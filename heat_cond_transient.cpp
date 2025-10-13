/*
To compile:
mpicxx -O3 -std=c++17 -fopenmp -I../mfem-4.8/ -I../hypre/src/hypre/include heat_cond_transient.cpp -o heat_cond_transient -L../mfem-4.8 -lmfem -L../hypre/src/hypre/lib -lHYPRE -L../metis-4.0 -lmetis -lrt -lopenblas

To run:
./heat_cond_transient -m mesh_files/through_hole.msh -tf 100 -dt 0.01 -nr 10

To visualise:
./../glvis-4.4/glvis -m heat_cond_transient_solns/cylinder_mfem_mesh.000000 -g heat_cond_transient_solns/cylinder_parallel_solution.000000
*/
//========================================================================
#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <chrono>
#include <tuple>
#include <filesystem>

using namespace std;
using namespace mfem;
using namespace chrono;
namespace fs = std::filesystem;
// Parallel Heat equation: du/dt = kappa * Laplacian(u)
class ParHeatConductionOperator : public TimeDependentOperator
{
private:
    ParFiniteElementSpace &fespace;
    Array<int> ess_tdof_list;
    ParBilinearForm *M, *K;
    HypreParMatrix *M_mat, *K_mat;
    HypreParVector M_lump;

    double kappa;
    double rho_cp;

    mutable HypreParVector z;
    // Optimized implicit solve members
    HypreParMatrix *A_mat;     // Pre-allocated system matrix
    HypreParVector *b_vec;     // Pre-allocated RHS vector
    HypreParVector *u_new_vec; // Pre-allocated solution vector
    HypreBoomerAMG *amg;       // Persistent AMG preconditioner
    HyprePCG *pcg;             // Persistent PCG solver
    double current_dt;         // Track current dt to avoid rebuilding
    bool solver_setup;         // Track if solver is set up
                               // Matrix-free option
    bool use_matrix_free;
    ParBilinearForm *K_mf;
    Array<int> conv_attr;
    Array<int> flux_attr;
    double h_conv;
    double T_amb;
    double q_flux;
    HypreParVector *f_vec;

public:
    double t1 = 0, t2 = 0, t3 = 0;
    ParHeatConductionOperator(ParFiniteElementSpace &f, double kappa, double rho_cp,
                              const Array<int> &ess_bdr,
                              const Array<int> &conv_bdr, double h, double Tamb,
                              const Array<int> &flux_bdr, double q,
                              bool matrix_free = false);
    virtual void Mult(const Vector &u, Vector &du_dt);
    void MatrixFreeMult(const Vector &u, Vector &du_dt);
    void ImplicitSolve(const double dt, const Vector &u, Vector &du_dt);
    void SetupImplicitSolver(const double dt);
    virtual ~ParHeatConductionOperator();
};
ParHeatConductionOperator::ParHeatConductionOperator(ParFiniteElementSpace &f,
                                                     double kappa, double rho_cp,
                                                     const Array<int> &ess_bdr,
                                                     const Array<int> &conv_bdr, double h, double Tamb,
                                                     const Array<int> &flux_bdr, double q,
                                                     bool matrix_free)
    : TimeDependentOperator(f.GetTrueVSize(), 0.0), fespace(f),
      kappa(kappa), rho_cp(rho_cp), z(&f), M_lump(&f), use_matrix_free(matrix_free),
      A_mat(nullptr), b_vec(nullptr), u_new_vec(nullptr),
      amg(nullptr), pcg(nullptr), current_dt(-1.0), solver_setup(false),
      conv_attr(conv_bdr), flux_attr(flux_bdr), h_conv(h), T_amb(Tamb), q_flux(q), f_vec(nullptr),
      M(nullptr), K(nullptr), M_mat(nullptr), K_mat(nullptr), K_mf(nullptr)
{
    fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
    // Linear form for boundary conditions
    ParLinearForm *f_lin = new ParLinearForm(&fespace);
    if (conv_attr.Sum() > 0)
    {
        ConstantCoefficient amb_coeff(h * Tamb);
        f_lin->AddBoundaryIntegrator(new BoundaryLFIntegrator(amb_coeff), conv_attr);
    }
    if (flux_attr.Sum() > 0)
    {
        ConstantCoefficient flux_coeff(q);
        f_lin->AddBoundaryIntegrator(new BoundaryLFIntegrator(flux_coeff), flux_attr);
    }
    f_lin->Assemble();
    f_vec = f_lin->ParallelAssemble();
    delete f_lin;
    // Mass matrix (always needed)
    M = new ParBilinearForm(&fespace);

    // Mass matrix = rho * cp * ∫ u v
    ConstantCoefficient rho_cp_coeff(rho_cp);
    M->AddDomainIntegrator(new MassIntegrator(rho_cp_coeff));

    M->Assemble();
    M->Finalize();
    M_mat = M->ParallelAssemble();
    // Compute lumped mass
    HypreParVector ones(&fespace);
    ones = 1.0;
    M_mat->Mult(ones, M_lump);
    if (use_matrix_free)
    {
        K_mf = new ParBilinearForm(&fespace);

        K_mf->AddDomainIntegrator(new DiffusionIntegrator());
        if (conv_attr.Sum() > 0)
        {
            ConstantCoefficient h_coeff(h);
            K_mf->AddBoundaryIntegrator(new BoundaryMassIntegrator(h_coeff), conv_attr);
        }
        K_mat = nullptr;
    }
    else
    {
        K = new ParBilinearForm(&fespace);
        // Stiffness = kappa * ∫ ∇u · ∇v
        ConstantCoefficient kappa_coeff(kappa);
        K->AddDomainIntegrator(new DiffusionIntegrator(kappa_coeff));
        if (conv_attr.Sum() > 0)
        {
            ConstantCoefficient h_coeff(h);
            K->AddBoundaryIntegrator(new BoundaryMassIntegrator(h_coeff), conv_attr);
        }
        K->Assemble();
        K->Finalize();
        K_mat = K->ParallelAssemble();
    }
    // Pre-allocate vectors for implicit solve
    b_vec = new HypreParVector(&fespace);
    u_new_vec = new HypreParVector(&fespace);
}
void ParHeatConductionOperator::Mult(const Vector &u, Vector &du_dt)
{
    // Compute: du_dt = M_lump^{-1}  (f - K  u)
    // Apply stiffness matrix: z = K * u
    auto start1 = high_resolution_clock::now();
    K_mat->Mult(u, z);
    z *= -1.0;          // Now z = -K u
    z.Add(1.0, *f_vec); // z = -K u + f
    auto end1 = high_resolution_clock::now();
    t1 = duration_cast<microseconds>(end1 - start1).count();
    // Apply lumped mass inverse (element-wise division)
    auto start2 = high_resolution_clock::now();
    for (int i = 0; i < z.Size(); i++)
    {
        if (fabs(M_lump[i]) > 1e-12)
        {
            du_dt[i] = z[i] / M_lump[i];
        }
        else
        {
            du_dt[i] = 0.0;
        }
    }
    auto end2 = high_resolution_clock::now();
    t2 = duration_cast<microseconds>(end2 - start2).count();
    // Zero out essential DOFs (boundary conditions don't change)
    auto start3 = high_resolution_clock::now();
    for (int i = 0; i < ess_tdof_list.Size(); i++)
    {
        du_dt[ess_tdof_list[i]] = 0.0;
    }
    auto end3 = high_resolution_clock::now();
    t3 = duration_cast<microseconds>(end3 - start3).count();
}
void ParHeatConductionOperator::SetupImplicitSolver(const double dt)
{
    if (fabs(dt - current_dt) < 1e-12 && solver_setup)
        return; // Already set up for this dt
                // Clean up previous solver if it exists
    if (A_mat)
        delete A_mat;
    if (amg)
        delete amg;
    if (pcg)
        delete pcg;
    // Create system matrix: A = M + dt*K (only when dt changes)
    A_mat = Add(1.0, *M_mat, dt, *K_mat);
    // Setup AMG preconditioner with optimized parameters
    amg = new HypreBoomerAMG();
    amg->SetPrintLevel(0);
    amg->SetMaxIter(1);            // Single V-cycle
    amg->SetTol(0.0);              // Don't solve exactly in preconditioner
    amg->SetMaxLevels(10);         // Limit levels for speed
    amg->SetStrongThresholdR(0.6); // Faster setup
                                   // Setup PCG solver with aggressive parameters
    pcg = new HyprePCG(MPI_COMM_WORLD);
    pcg->SetTol(1e-6);     // Looser tolerance for speed
    pcg->SetMaxIter(1000); // Fewer iterations
    pcg->SetPrintLevel(0);
    pcg->SetPreconditioner(*amg);
    pcg->SetOperator(*A_mat);
    current_dt = dt;
    solver_setup = true;
}
void ParHeatConductionOperator::ImplicitSolve(const double dt, const Vector &u, Vector &du_dt)
{
    // computes du_dt, does not modify u directly.

    // Setup solver only if dt changed (major optimization)
    SetupImplicitSolver(dt);

    // Right hand side: b = M * u + dt * f
    // Backward Euler: (M+dtK)*u[n+1] = M*u[n] + dt*f
    M_mat->Mult(u, *b_vec);
    b_vec->Add(dt, *f_vec);

    // A=M+dt*K is the system matrix
    // Solve A * u_new = b (reuse pre-allocated vector)
    pcg->Mult(*b_vec, *u_new_vec);

    // Compute du_dt = (u_new - u_old) / dt
    // Vectorized operation for better performance
    const double inv_dt = 1.0 / dt;
    const int size = u_new_vec->Size();
    const double *u_data = u.GetData();
    const double *u_new_data = u_new_vec->GetData();
    double *du_dt_data = du_dt.GetData();
// Use compiler vectorization hints
#pragma omp simd
    for (int i = 0; i < size; i++)
    {
        du_dt_data[i] = (u_new_data[i] - u_data[i]) * inv_dt;
    }
    // Zero out essential DOFs (vectorized)
    const int ess_size = ess_tdof_list.Size();
    const int *ess_data = ess_tdof_list.GetData();
#pragma omp simd
    for (int i = 0; i < ess_size; i++)
    {
        du_dt_data[ess_data[i]] = 0.0;
    }
}

ParHeatConductionOperator::~ParHeatConductionOperator()
{
    delete M_mat;
    delete K_mat;
    delete M;
    delete K;
    delete K_mf;
    delete f_vec;
}

// Smooth quadratic initial temperature distribution
// T(z) = T_end - (T_end - T_center) * (2z/L - 1)^2
// where z is the axial coordinate, L is cylinder length
class QuadraticInitialTemperature : public Coefficient
{
private:
    double T_end;        // Temperature at ends (100°C)
    double T_center;     // Temperature at center (can be different from ambient)
    double L;            // Cylinder length
    double z_min, z_max; // Cylinder bounds in z-direction
    int axis_dir;        // Which coordinate axis is the cylinder axis (0=x, 1=y, 2=z)

public:
    QuadraticInitialTemperature(double end_temp, double center_temp, double length,
                                double z_minimum, double z_maximum, int axis = 2)
        : T_end(end_temp), T_center(center_temp), L(length),
          z_min(z_minimum), z_max(z_maximum), axis_dir(axis) {}

    virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip)
    {
        Vector x(3);
        T.Transform(ip, x);

        // Get coordinate along cylinder axis
        double z_coord = x[axis_dir];

        // Normalize to [-1, 1] range
        double xi = 2.0 * (z_coord - z_min) / (z_max - z_min) - 1.0;

        // Quadratic profile: T(xi) = T_end - (T_end - T_center) * xi^2
        double temperature = T_end - (T_end - T_center) * xi * xi;

        return temperature;
    }
};

// Alternative: Cosine-based smooth profile (even smoother)
class CosineInitialTemperature : public Coefficient
{
private:
    double T_end;
    double T_center;
    double z_min, z_max;
    int axis_dir;

public:
    CosineInitialTemperature(double end_temp, double center_temp,
                             double z_minimum, double z_maximum, int axis = 2)
        : T_end(end_temp), T_center(center_temp),
          z_min(z_minimum), z_max(z_maximum), axis_dir(axis) {}

    virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip)
    {
        Vector x(3);
        T.Transform(ip, x);

        double z_coord = x[axis_dir];
        double xi = (z_coord - z_min) / (z_max - z_min); // Normalize to [0, 1]

        // Cosine profile: T(xi) = T_center + (T_end - T_center) * cos(π*xi)
        double temperature = T_center + (T_end - T_center) * cos(M_PI * xi);

        return temperature;
    }
};

int main(int argc, char *argv[])
{
    // Initialize MPI and Hypre
    Mpi::Init(argc, argv);
    Hypre::Init();
    int num_procs = Mpi::WorldSize();
    int myid = Mpi::WorldRank();
    const char *mesh_file = "cylinder_3d.msh";
    int order = 1;
    double t_final = 500.0;
    double dt = 0.001;
    const char *mfem_mesh = "cylinder_mfem_mesh";
    const char *transient_sol = "cylinder_transient_solution";
    string output_dir = "heat_cond_transient_solns";

    // Material properties (e.g. Aluminum)
    double kappa = 1000000.0; // thermal conductivity [W/m-K]
    double rho = 2700.0;      // density [kg/m^3]
    double cp = 900.0;        // specific heat [J/kg-K]
    double rho_cp = rho * cp; // volumetric heat capacity [J/m^3-K]

    int vis_steps = 10;
    bool visualization = true;
    bool use_implicit = true;
    int precision = 8;
    int n_report = 100;
    OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
    args.AddOption(&order, "-o", "--order", "Finite element order.");
    args.AddOption(&t_final, "-tf", "--t-final", "Final time.");
    args.AddOption(&dt, "-dt", "--time-step", "Time step size.");
    args.AddOption(&vis_steps, "-vs", "--visualization-steps", "Visualize every n-th timestep.");
    args.AddOption(&use_implicit, "-imp", "--implicit", "-no-imp", "--explicit",
                   "Use implicit (backward Euler) time integration for stability.");
    args.AddOption(&n_report, "-nr", "--n_report", "Print timings every n_report time steps");
    args.Parse();
    if (!args.Good())
    {
        if (myid == 0)
        {
            args.PrintUsage(cout);
        }
        return 1;
    }
    if (myid == 0)
    {
        args.PrintOptions(cout);
    }
    // Read mesh and partition it
    Mesh *mesh = new Mesh(mesh_file, 1, 1);
    int dim = mesh->Dimension();
    if (myid == 0)
    {
        cout << "Mesh dimension: " << dim << endl;
        cout << "Number of processors: " << num_procs << endl;
    }
    // Parallel mesh partitioning
    ParMesh pmesh(MPI_COMM_WORLD, *mesh);
    delete mesh; // The parallel mesh now owns the serial mesh data
    // Define parallel finite element space
    H1_FECollection fec(order, dim);
    ParFiniteElementSpace *pfespace = new ParFiniteElementSpace(&pmesh, &fec);
    HYPRE_BigInt size = pfespace->GlobalTrueVSize();
    if (myid == 0)
    {
        cout << "Number of finite element unknowns: " << size << endl;
    }
    // Set up boundary conditions
    Array<int> ess_bdr(pmesh.bdr_attributes.Max());
    ess_bdr = 0;
    // // Apply essential BCs only to flat ends
    // if (pmesh.bdr_attributes.Max() >= 3)
    // {
    //     ess_bdr[2] = 1;
    //     ess_bdr[3] = 1;
    // }
    Array<int> conv_bdr(pmesh.bdr_attributes.Max());
    conv_bdr = 0;
    if (pmesh.bdr_attributes.Max() >= 2)
    {
        conv_bdr[1] = 1; // face 1, attribute 2
    }
    Array<int> flux_bdr(pmesh.bdr_attributes.Max());
    flux_bdr = 0;
    if (pmesh.bdr_attributes.Max() >= 1)
    {
        flux_bdr[0] = 1; // face 0, attribute 1
    }

    double h_conv_val = 20.0;
    double T_amb_val = 10.0;
    double q_val = 1000000.0;
    if (myid == 0)
    {
        cout << "Essential boundary conditions applied to attributes: ";
        for (int i = 0; i < ess_bdr.Size(); i++)
        {
            if (ess_bdr[i])
                cout << i + 1 << " ";
        }
        cout << endl;
    }

    // Get cylinder bounds for smooth initial condition
    double z_min = 1e10, z_max = -1e10;

    // Find z-bounds of the mesh (assuming cylinder axis is z-direction)
    for (int i = 0; i < pmesh.GetNV(); i++)
    {
        double *vertex = pmesh.GetVertex(i);
        z_min = std::min(z_min, vertex[2]); // Assuming z is axis 2
        z_max = std::max(z_max, vertex[2]);
    }

    // Reduce to get global bounds across all processors
    double z_min_global, z_max_global;
    MPI_Allreduce(&z_min, &z_min_global, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&z_max, &z_max_global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    double cylinder_length = z_max_global - z_min_global;

    if (myid == 0)
    {
        cout << "Cylinder bounds: z ∈ [" << z_min_global << ", " << z_max_global
             << "], length = " << cylinder_length << endl;
    }

    // Initialize with smooth quadratic temperature profile
    ParGridFunction u_gf(pfespace);
    u_gf = T_amb_val;
    // Choose your preferred profile:
    // Option 1: Quadratic profile (100°C at ends, 60°C at center)
    // QuadraticInitialTemperature quad_temp(100.0, 0.0, cylinder_length,
    // z_min_global, z_max_global, 2);

    // Option 2: Cosine profile (smoother, 100°C at ends, 10°C at center)
    // CosineInitialTemperature cos_temp(100.0, 10.0, z_min_global, z_max_global, 2);

    // Project the smooth initial condition
    // u_gf.ProjectCoefficient(quad_temp);
    // u_gf.ProjectCoefficient(cos_temp); // Use this for cosine profile

    // Still need to enforce boundary conditions after projection
    // Array<int> attr_hot(pmesh.bdr_attributes.Max());
    // attr_hot = 0;
    // if (pmesh.bdr_attributes.Max() >= 3)
    // {
    //     attr_hot[2] = 1;
    //     attr_hot[3] = 1;
    // }

    // // Apply boundary conditions to ensure exact values at boundaries
    // Array<int> ess_tdof_list_hot;
    // if (attr_hot.Max() > 0)
    // {
    //     pfespace->GetEssentialTrueDofs(attr_hot, ess_tdof_list_hot);
    //     for (int i = 0; i < ess_tdof_list_hot.Size(); i++)
    //     {
    //         u_gf(ess_tdof_list_hot[i]) = 0.0;
    //     }
    // }

    // Get parallel true DOF vector
    HypreParVector u(pfespace);
    u_gf.GetTrueDofs(u);

    if (myid == 0)
    {
        cout << "Smooth initial conditions set:" << endl;
        cout << "Profile: Quadratic with 100°C at ends" << endl;
        // cout << "Hot boundary DOFs: " << ess_tdof_list_hot.Size() << endl;
        cout << "Max initial temperature: " << u_gf.Max() << "°C" << endl;
        cout << "Min initial temperature: " << u_gf.Min() << "°C" << endl;
    }

    // Initialize parallel heat conduction operator
    ParHeatConductionOperator oper(*pfespace, kappa, rho_cp, ess_bdr, conv_bdr, h_conv_val, T_amb_val, flux_bdr, q_val);
    // Calculate stable time step based on mesh and alpha
    double h_min = pmesh.GetElementSize(0);
    for (int i = 1; i < pmesh.GetNE(); i++)
    {
        h_min = min(h_min, pmesh.GetElementSize(i));
    }
    // Reduce to get global minimum across all processors
    double h_min_global;
    MPI_Allreduce(&h_min, &h_min_global, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    h_min = h_min_global;
    // Stability condition: dt <= h^2 / (2  alpha  dim)
    // double dt_stable = h_min * h_min / (2.0 * alpha * dim);
    // if (dt > dt_stable)
    // {
    //     if (myid == 0)
    //     {
    //         cout << "Warning: Requested dt = " << dt << " is unstable!" << endl;
    //         cout << "Maximum stable dt = " << dt_stable << " (based on mesh size h_min = " << h_min << ")" << endl;
    //         cout << "Using stable time step: " << dt_stable << endl;
    //     }
    //     dt = dt_stable * 0.8;
    // }
    if (myid == 0)
    {
        cout << "Mesh minimum element size: " << h_min << endl;
        // cout << "Stable time step limit: " << dt_stable << endl;
        cout << "Using " << (use_implicit ? "implicit" : "explicit") << " time integration" << endl;
    }
    // Time integration
    double t = 0.0;
    int step = 0;
    HypreParVector du_dt(pfespace);
    if (myid == 0)
    {
        cout << "\nStarting parallel time integration..." << endl;
        cout << "Time step: " << dt << ", Final time: " << t_final << endl;
    }
    // Timing variables (per processor)
    double time_implicit = 0.0, time_explicit = 0.0, time_bc = 0.0;
    double time_clip = 0.0, time_bounds_check = 0.0, time_visualization = 0.0;
    double time_loop_total = 0.0;
    double t1 = 0.0, t2 = 0.0, t3 = 0.0;
    while (t < t_final - 1e-8 * dt)
    {
        auto loop_start = high_resolution_clock::now();
        step++;
        double dt_real = std::min(dt, t_final - t);
        // Implicit or explicit time stepping
        if (use_implicit)
        {
            auto t0 = high_resolution_clock::now();

            // Solve equation to compute du/dt
            oper.ImplicitSolve(dt_real, u, du_dt);

            // Update u as u = u + dt* du__dt
            add(u, dt_real, du_dt, u);
            auto step_end = high_resolution_clock::now();
            time_implicit += duration_cast<microseconds>(step_end - t0).count();
        }
        else
        {
            auto t0 = high_resolution_clock::now();
            oper.Mult(u, du_dt);
            t1 += oper.t1;
            t2 += oper.t2;
            t3 += oper.t3;

            // Check for instability (reduce across all processors)
            double max_du_dt_local = 0.0;
            for (int i = 0; i < du_dt.Size(); i++)
                max_du_dt_local = std::max(max_du_dt_local, fabs(du_dt[i]));
            double max_du_dt_global;

            MPI_Allreduce(&max_du_dt_local, &max_du_dt_global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            if (max_du_dt_global > 1e6)
            {
                if (myid == 0)
                    std::cout << "Instability detected!\n";
                Mpi::Finalize();
                return 1;
            }
            add(u, dt_real, du_dt, u);
            auto step_end = high_resolution_clock::now();
            time_explicit += duration_cast<microseconds>(step_end - t0).count();
        }
        // Apply boundary conditions
        auto t_bc0 = high_resolution_clock::now();
        // if (attr_hot.Max() > 0)
        //     for (int i = 0; i < ess_tdof_list_hot.Size(); i++)
        //         u[ess_tdof_list_hot[i]] = 0.0;
        // if (attr_cold.Max() > 0)
        //     for (int i = 0; i < ess_tdof_list_cold.Size(); i++)
        //         u[ess_tdof_list_cold[i]] = 0.0;
        auto t_bc1 = high_resolution_clock::now();
        time_bc += duration_cast<microseconds>(t_bc1 - t_bc0).count();
        // Bounds check
        auto t_chk0 = high_resolution_clock::now();
        u_gf.SetFromTrueDofs(u);
        double max_temp = u_gf.Max();
        double min_temp = u_gf.Min();
        auto t_chk1 = high_resolution_clock::now();
        time_bounds_check += duration_cast<microseconds>(t_chk1 - t_chk0).count();
        t += dt_real;
        auto loop_end = high_resolution_clock::now();
        time_loop_total += duration_cast<microseconds>(loop_end - loop_start).count();
        // Print profiling info (only from processor 0)
        if ((step % n_report == 0 || t >= t_final - 1e-8 * dt) && myid == 0)
        {
            std::cout << "\n[Profiling up to step " << step << " - Processor 0]\n";
            std::cout << " Loop total: " << time_loop_total / 1e6 << " s\n";
            if (use_implicit)
                std::cout << " Implicit solve: " << time_implicit / 1e6 << " s\n";
            else
            {
                std::cout << " Explicit step: " << time_explicit / 1e6 << " s\n";
                std::cout << "\tt1: " << t1 / 1e6 << " s\n";
                std::cout << "\tt2: " << t2 / 1e6 << " s\n";
                std::cout << "\tt3: " << t3 / 1e6 << " s\n";
            }
            std::cout << " Boundary cond.: " << time_bc / 1e6 << " s\n";
            std::cout << " Clipping: " << time_clip / 1e6 << " s\n";
            std::cout << " Bounds check: " << time_bounds_check / 1e6 << " s\n";
            std::cout << " Max temp for proc : " << myid << "=" << max_temp << ", Min temp for proc " << myid << "=" << min_temp << "\n\n";
        }
    }
    // Final results (reduce across processors for global min/max)
    u_gf.SetFromTrueDofs(u);
    double max_temp_local = u_gf.Max();
    double min_temp_local = u_gf.Min();
    double max_temp_global, min_temp_global;
    MPI_Allreduce(&max_temp_local, &max_temp_global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&min_temp_local, &min_temp_global, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    if (myid == 0)
    {
        cout << "\n=== Final Results ===" << endl;
        cout << "Simulation completed at time: " << t << "s" << endl;
        cout << "Global max temperature: " << max_temp_global << "°C" << endl;
        cout << "Global min temperature: " << min_temp_global << "°C" << endl;
    }
    // Save solution (each processor saves its part)
    if (myid == 0)
    {
        fs::create_directories(output_dir);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    {
        ostringstream mesh_name, sol_name;
        mesh_name << output_dir << "/" << mfem_mesh << "." << setfill('0') << setw(6) << myid;
        sol_name << output_dir << "/" << transient_sol << "." << setfill('0') << setw(6) << myid;

        ofstream mesh_ofs(mesh_name.str().c_str());
        mesh_ofs.precision(precision);
        pmesh.Print(mesh_ofs);

        ofstream sol_ofs(sol_name.str().c_str());
        sol_ofs.precision(precision);
        u_gf.Save(sol_ofs);
    }
    if (myid == 0)
    {
        cout << "\nParallel solution saved to cylinder_parallel_solution.xxxxxx files" << endl;
        cout << "Parallel mesh saved to cylinder_parallel_mesh.xxxxxx files" << endl;
        cout << "\nNote: For full steady-state, you may need longer simulation time." << endl;
        cout << "The parallel computation distributes the mesh across " << num_procs << " processors." << endl;
    }
    // Cleanup
    delete pfespace;
    Mpi::Finalize();
    return 0;
}