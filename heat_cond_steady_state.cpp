/*
To compile:
mpicxx -O3 -std=c++17 -fopenmp -I../mfem-4.8/ -I../hypre/src/hypre/include heat_cond_steady_state.cpp -o heat_cond_steady_state -L../mfem-4.8 -lmfem -L../hypre/src/hypre/lib -lHYPRE -L../metis-4.0 -lmetis -lrt -lopenblas

To run:
./heat_cond_steady_state -m mesh_files/two_holes_ref1.msh

To visualise:
./../glvis-4.4/glvis -m heat_cond_steady_state_solns/cylinder_mfem_mesh.000000 -g heat_cond_steady_state_solns/cylinder_steady_solution.000000
*/
//========================================================================
#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <filesystem>

using namespace std;
using namespace mfem;
namespace fs = std::filesystem;

int main(int argc, char *argv[])
{
    // 1. Initialize MPI and HYPRE.
    Mpi::Init(argc, argv);
    int num_procs = Mpi::WorldSize();
    int myid = Mpi::WorldRank();
    Hypre::Init();

    // 2. Parse command-line options.
    const char *mesh_file = "cylinder_3d.msh";
    int order = 1;
    double kappa = 205.0; // Represents kappa in this simplified code
    double alpha = 0.01;
    double heat_flux = 1000.0; // Heat flux value (W/m^2 or similar units)
    double h_conv = 20.0;      // Convective heat transfer coefficient (W/m^2·K)
    double T_ambient = 25.0;   // Ambient temperature (°C or K)
    int precision = 8;
    int max_iter = 100; // Max Picard iterations
    double tol = 1e-6;  // Convergence tolerance
    const char *mfem_mesh = "cylinder_mfem_mesh";
    const char *steady_sol = "cylinder_steady_solution";
    string output_dir = "heat_cond_steady_state_solns";

    OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
    args.AddOption(&order, "-o", "--order", "Finite element order.");
    args.AddOption(&kappa, "-k", "--kappa", "Conductivity coeff");
    args.AddOption(&alpha, "-a", "--alpha", "Multiplied with u, result added to kappa");
    args.AddOption(&heat_flux, "-q", "--heat-flux", "Heat flux value for boundary condition");
    args.AddOption(&h_conv, "-h", "--h-conv", "Convective heat transfer coefficient");
    args.AddOption(&T_ambient, "-ta", "--t-ambient", "Ambient temperature for convection");
    args.AddOption(&max_iter, "-mi", "--max-iter", "Maximum number of Picard iterations.");
    args.AddOption(&tol, "-tol", "--tolerance", "Convergence tolerance for Picard iterations.");
    args.AddOption(&mfem_mesh, "-sm", "--mfem_mesh", "Steady state mesh");
    args.AddOption(&steady_sol, "-sol", "--steady_sol", "Steady state solution");
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

    // 3. Read the (serial) mesh from the given file on all processors.
    Mesh *mesh = new Mesh(mesh_file, 1, 1);
    int dim = mesh->Dimension();

    // 4. Create a parallel mesh by partitioning the serial mesh.
    ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
    delete mesh;

    // 5. Define a parallel finite element space on the parallel mesh.
    H1_FECollection fec(order, dim);
    ParFiniteElementSpace *pfespace = new ParFiniteElementSpace(pmesh, &fec);
    HYPRE_BigInt size = pfespace->GlobalTrueVSize();
    if (myid == 0)
    {
        cout << "Number of finite element unknowns: " << size << endl;
    }

    // 6. Define the Dirichlet boundary conditions.
    // The problem has:
    // - Attribute 1: Curved surface with convective BC (natural BC)
    // - Attribute 2: Hot end with heat flux BC (natural BC)
    // - Attribute 3: Cold end with fixed temperature (Dirichlet BC)
    Array<int> ess_tdof_list;
    Array<int> ess_bdr(pmesh->bdr_attributes.Max());
    ess_bdr = 0;
    // Only attribute 3: Cold end (0 C) - Dirichlet BC
    if (pmesh->bdr_attributes.Max() >= 3)
    {
        ess_bdr[3] = 1;
        ess_bdr[4] = 1; // attribute 3 (cold end)
    }
    pfespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

    // 7. Set up the solution grid function, u.
    ParGridFunction u_gf(pfespace);
    u_gf = T_ambient; // Initial guess for the temperature is 0.

    // 8. Apply the Dirichlet boundary condition values to the grid function.
    // Only apply to the cold end now
    ConstantCoefficient cold_temp(100.0);

    Array<int> cold_bdr(pmesh->bdr_attributes.Max());
    cold_bdr = 0;
    if (pmesh->bdr_attributes.Max() >= 3)
    {
        cold_bdr[3] = 1; // Attribute 3 (cold end)
        cold_bdr[4] = 1; // Attribute 3 (cold end)
    }
    u_gf.ProjectBdrCoefficient(cold_temp, cold_bdr);

    // ---------------------------------------------------------------
    // --- START OF STEADY-STATE SOLVER SECTION ---
    // ---------------------------------------------------------------

    if (myid == 0)
    {
        cout << "\nStarting steady-state solver with Picard iteration..." << endl;
        cout << "Heat flux boundary condition: " << heat_flux << " applied to boundary attribute 2" << endl;
        cout << "Convective boundary: h = " << h_conv << ", T_ambient = " << T_ambient << " applied to boundary attribute 1" << endl;
    }

    // Create a HypreParVector for the previous iteration's solution
    HypreParVector *u_prev = u_gf.GetTrueDofs();
    HypreParVector u_change(pfespace);

    // Picard iteration loop
    for (int iter = 1; iter <= max_iter; ++iter)
    {
        // a. Define the nonlinear conductivity coefficient: k = (kappa + alpha*u)
        // Here, we use the value of u from the previous step (u_gf).
        ConstantCoefficient kappa_coeff(kappa);
        ConstantCoefficient alpha_coeff(alpha);
        GridFunctionCoefficient u_coeff(&u_gf);
        ProductCoefficient alpha_u_coeff(alpha_coeff, u_coeff);
        SumCoefficient k_coeff(kappa_coeff, alpha_u_coeff);

        // b. Set up the bilinear form a(u) = k * grad(u) . grad(v)
        ParBilinearForm a(pfespace);
        a.AddDomainIntegrator(new DiffusionIntegrator(k_coeff));

        // Add convective boundary condition to bilinear form
        // Convection contributes: ∫_Γ h * u * v dΓ
        ConstantCoefficient h_coeff(h_conv);
        Array<int> conv_bdr(pmesh->bdr_attributes.Max());
        conv_bdr = 0;
        if (pmesh->bdr_attributes.Max() >= 1)
        {
            conv_bdr[1] = 1; // attribute 1 (curved surface with convection)
            conv_bdr[2] = 1; // attribute 1 (curved surface with convection)
        }
        a.AddBoundaryIntegrator(new MassIntegrator(h_coeff), conv_bdr);

        a.Assemble();

        // c. Set up the linear form b(v) with heat flux and convective boundary conditions
        ParLinearForm b(pfespace);
        b = 0.0;

        // Add heat flux boundary condition on attribute 2
        // The weak form contribution is: ∫_Γ q * v dΓ where q is the heat flux
        ConstantCoefficient heat_flux_coeff(heat_flux);
        Array<int> flux_bdr(pmesh->bdr_attributes.Max());
        flux_bdr = 0;
        if (pmesh->bdr_attributes.Max() >= 2)
        {
            flux_bdr[0] = 1; // attribute 2 (heat flux boundary)
        }
        b.AddBdrFaceIntegrator(new BoundaryLFIntegrator(heat_flux_coeff), flux_bdr);

        // Add convective boundary condition to linear form
        // Convection contributes: ∫_Γ h * T_ambient * v dΓ
        ConstantCoefficient h_T_amb_coeff(h_conv * T_ambient);
        b.AddBdrFaceIntegrator(new BoundaryLFIntegrator(h_T_amb_coeff), conv_bdr);

        b.Assemble();

        // d. Form the linear system A*U = B.
        HypreParMatrix A;
        Vector B, U;
        a.FormLinearSystem(ess_tdof_list, u_gf, b, A, U, B);

        // e. Solve the linear system using PCG with AMG preconditioner.
        HypreBoomerAMG amg(A);
        HyprePCG pcg(A);
        pcg.SetTol(1e-12);
        pcg.SetMaxIter(200);
        pcg.SetPrintLevel(0);
        pcg.SetPreconditioner(amg);
        pcg.Mult(B, U);

        // f. Update the grid function with the new solution.
        a.RecoverFEMSolution(U, b, u_gf);

        // g. Check for convergence.
        u_change.Set(1.0, *u_gf.GetTrueDofs());
        u_change -= *u_prev;
        double change_norm = u_change.Norml2();
        double u_norm = u_gf.GetTrueDofs()->Norml2();

        // Store current solution for the next iteration's convergence check
        *u_prev = *u_gf.GetTrueDofs();

        if (myid == 0)
        {
            cout << "Iteration " << iter << ": |u_new - u_old|/|u_new| = "
                 << change_norm / u_norm << endl;
        }

        if (change_norm / u_norm < tol)
        {
            if (myid == 0)
            {
                cout << "Convergence reached!" << endl;
            }
            break;
        }
    }

    delete u_prev;

    // ---------------------------------------------------------------
    // --- END OF STEADY-STATE SOLVER SECTION ---
    // ---------------------------------------------------------------
    if (myid == 0)
    {
        fs::create_directories(output_dir);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    // 9. Save the final solution.
    {
        ostringstream mesh_name, sol_name;
        mesh_name << output_dir << "/" << mfem_mesh << "." << setfill('0') << setw(6) << myid;
        sol_name << output_dir << "/" << steady_sol << "." << setfill('0') << setw(6) << myid;

        ofstream mesh_ofs(mesh_name.str().c_str());
        mesh_ofs.precision(precision);
        pmesh->Print(mesh_ofs);

        ofstream sol_ofs(sol_name.str().c_str());
        sol_ofs.precision(precision);
        u_gf.Save(sol_ofs);
    }

    if (myid == 0)
    {
        cout << "\nSteady-state solution saved." << endl;
    }

    // 10. Clean up.
    delete pfespace;
    delete pmesh;

    Mpi::Finalize();
    return 0;
}