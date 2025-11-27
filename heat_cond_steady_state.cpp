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
#include <map>
#include <set>

using namespace std;
using namespace mfem;
namespace fs = std::filesystem;

// Helper function to identify boundary attributes based on geometry
void IdentifyBoundaryAttributes(ParMesh *pmesh, int &curved_attr, int &top_attr, int &bottom_attr, int myid)
{
    // Get the bounding box of the mesh
    Vector bbmin, bbmax;
    pmesh->GetBoundingBox(bbmin, bbmax);

    double z_min = bbmin(2);
    double z_max = bbmax(2);
    double z_tol = 0.05 * (z_max - z_min); // 5% tolerance

    if (myid == 0)
    {
        cout << "\nMesh bounding box:" << endl;
        cout << "  X: [" << bbmin(0) << ", " << bbmax(0) << "]" << endl;
        cout << "  Y: [" << bbmin(1) << ", " << bbmax(1) << "]" << endl;
        cout << "  Z: [" << bbmin(2) << ", " << bbmax(2) << "]" << endl;
        cout << "  Z tolerance: " << z_tol << endl;
    }

    // Map to store: attribute -> {avg_z, face_count}
    map<int, pair<double, int>> attr_info;

    // Scan all boundary elements
    for (int i = 0; i < pmesh->GetNBE(); i++)
    {
        int attr = pmesh->GetBdrAttribute(i);
        Array<int> vertices;
        pmesh->GetBdrElementVertices(i, vertices);

        // Calculate average z-coordinate of this boundary face
        double z_sum = 0.0;
        for (int j = 0; j < vertices.Size(); j++)
        {
            double *coords = pmesh->GetVertex(vertices[j]);
            z_sum += coords[2];
        }
        double z_avg = z_sum / vertices.Size();

        // Update attribute info
        if (attr_info.find(attr) == attr_info.end())
        {
            attr_info[attr] = {z_avg, 1};
        }
        else
        {
            double prev_z = attr_info[attr].first;
            int prev_count = attr_info[attr].second;
            attr_info[attr].first = (prev_z * prev_count + z_avg) / (prev_count + 1);
            attr_info[attr].second = prev_count + 1;
        }
    }

    // Classify attributes
    curved_attr = -1;
    top_attr = -1;
    bottom_attr = -1;

    for (auto &entry : attr_info)
    {
        int attr = entry.first;
        double avg_z = entry.second.first;
        int count = entry.second.second;

        if (myid == 0)
        {
            cout << "Attribute " << attr << ": avg_z = " << avg_z
                 << ", faces = " << count << endl;
        }

        // Check if this is top, bottom, or curved surface
        if (fabs(avg_z - z_max) < z_tol)
        {
            top_attr = attr;
            if (myid == 0)
                cout << "  -> Identified as TOP surface" << endl;
        }
        else if (fabs(avg_z - z_min) < z_tol)
        {
            bottom_attr = attr;
            if (myid == 0)
                cout << "  -> Identified as BOTTOM surface" << endl;
        }
        else
        {
            curved_attr = attr;
            if (myid == 0)
                cout << "  -> Identified as CURVED surface" << endl;
        }
    }

    if (myid == 0)
    {
        cout << "\nBoundary classification:" << endl;
        cout << "  Curved surface (convection): attribute " << curved_attr << endl;
        cout << "  Top surface (heat flux):     attribute " << top_attr << endl;
        cout << "  Bottom surface (cold):       attribute " << bottom_attr << endl;
    }
}

int main(int argc, char *argv[])
{
    // 1. Initialize MPI and HYPRE.
    Mpi::Init(argc, argv);
    int num_procs = Mpi::WorldSize();
    int myid = Mpi::WorldRank();
    Hypre::Init();

    // 2. Parse command-line options.
    const char *mesh_file = "shell-gmsh2.gmsh2";
    int order = 1;
    double kappa = 205.0;    // Thermal conductivity (W/m·K) - Aluminum
    double alpha = 0.01;     // Temperature-dependent coefficient
    double heat_flux = 500.0;  // Heat flux value (W/m^2)
    double h_conv = 20.0;    // Convective heat transfer coefficient (W/m^2·K)
    double T_ambient = 25.0; // Ambient temperature (°C)
    double T_cold = 25.0;    // Cold end temperature (°C)
    int precision = 8;
    int max_iter = 100; // Max Picard iterations
    double tol = 1e-9;  // Convergence tolerance
    string output_dir = "heat_cond_steady_state_solns";

    OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
    args.AddOption(&order, "-o", "--order", "Finite element order.");
    args.AddOption(&kappa, "-k", "--kappa", "Thermal conductivity (W/m·K)");
    args.AddOption(&alpha, "-a", "--alpha", "Temperature-dependent coefficient");
    args.AddOption(&heat_flux, "-q", "--heat-flux", "Heat flux on top surface (W/m^2)");
    args.AddOption(&h_conv, "-h", "--h-conv", "Convective heat transfer coefficient (W/m^2·K)");
    args.AddOption(&T_ambient, "-ta", "--t-ambient", "Ambient temperature (°C)");
    args.AddOption(&T_cold, "-tc", "--t-cold", "Bottom surface temperature (°C)");
    args.AddOption(&max_iter, "-mi", "--max-iter", "Maximum Picard iterations");
    args.AddOption(&tol, "-tol", "--tolerance", "Convergence tolerance");
    // args.AddOption(&mfem_mesh, "-sm", "--mfem_mesh", "Output mesh name");
    // args.AddOption(&steady_sol, "-sol", "--steady_sol", "Output solution name");
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

    fs::path mesh_path(mesh_file);
    string base_name = mesh_path.stem().string();
    std::string mfem_mesh = base_name + "_mfem_mesh";
    std::string steady_sol = base_name + "_steady_solution";

    // 3. Read the mesh
    Mesh *mesh = new Mesh(mesh_file, 1, 1);
    int dim = mesh->Dimension();

    // 4. Create parallel mesh
    ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
    delete mesh;

    // 5. Identify boundary attributes automatically
    int curved_attr, top_attr, bottom_attr;
    IdentifyBoundaryAttributes(pmesh, curved_attr, top_attr, bottom_attr, myid);

    if (curved_attr == -1 || top_attr == -1 || bottom_attr == -1)
    {
        if (myid == 0)
        {
            cerr << "Error: Could not identify all boundary surfaces!" << endl;
        }
        delete pmesh;
        return 1;
    }

    // 6. Define finite element space
    H1_FECollection fec(order, dim);
    ParFiniteElementSpace *pfespace = new ParFiniteElementSpace(pmesh, &fec);
    HYPRE_BigInt size = pfespace->GlobalTrueVSize();
    if (myid == 0)
    {
        cout << "\nNumber of finite element unknowns: " << size << endl;
    }

    // 7. Define Dirichlet boundary conditions on bottom surface
    Array<int> ess_tdof_list;
    Array<int> ess_bdr(pmesh->bdr_attributes.Max());
    ess_bdr = 0;
    // ess_bdr[bottom_attr - 1] = 1; // Bottom surface has fixed temperature
    // pfespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

    // 8. Set up solution grid function
    ParGridFunction u_gf(pfespace);
    u_gf = T_ambient; // Initial guess

    // // 9. Apply Dirichlet BC on bottom surface
    // ConstantCoefficient cold_temp(T_cold);
    // Array<int> cold_bdr(pmesh->bdr_attributes.Max());
    // cold_bdr = 0;
    // cold_bdr[bottom_attr - 1] = 1;
    // u_gf.ProjectBdrCoefficient(cold_temp, cold_bdr);

    // ---------------------------------------------------------------
    // --- STEADY-STATE SOLVER WITH PICARD ITERATION ---
    // ---------------------------------------------------------------

    if (myid == 0)
    {
        cout << "\n"
             << string(60, '=') << endl;
        cout << "Starting steady-state solver with Picard iteration" << endl;
        cout << string(60, '=') << endl;
        cout << "Key faces:" << endl;
        cout << "  - Top surface : face " << top_attr << endl;
        cout << "  - Bottom surface : face " << bottom_attr << endl;
        cout << "  - Curved surface : face " << curved_attr << endl;
        cout << string(60, '=') << endl;
    }

    HypreParVector *u_prev = u_gf.GetTrueDofs();
    HypreParVector u_change(pfespace);

    // Picard iteration loop
    for (int iter = 1; iter <= max_iter; ++iter)
    {
        // a. Define nonlinear conductivity: k = kappa + alpha*u
        ConstantCoefficient kappa_coeff(kappa);
        ConstantCoefficient alpha_coeff(alpha);
        GridFunctionCoefficient u_coeff(&u_gf);
        ProductCoefficient alpha_u_coeff(alpha_coeff, u_coeff);
        SumCoefficient k_coeff(kappa_coeff, alpha_u_coeff);

        // b. Bilinear form: a(u,v) = ∫_Ω k·∇u·∇v dΩ + ∫_Γ_conv h·u·v dΓ
        ParBilinearForm a(pfespace);
        a.AddDomainIntegrator(new DiffusionIntegrator(k_coeff));

        // Add convection on curved surface
        ConstantCoefficient h_coeff(h_conv);
        Array<int> conv_bdr(pmesh->bdr_attributes.Max());
        conv_bdr = 1;               // Set all boundaries to have convection
        conv_bdr[top_attr - 1] = 0; // Exclude top surface (insulated/zero flux)
        a.AddBoundaryIntegrator(new MassIntegrator(h_coeff), conv_bdr);
        a.Assemble();

        // c. Linear form: b(v) = ∫_Γ_flux q·v dΓ + ∫_Γ_conv h·T_amb·v dΓ
        ParLinearForm b(pfespace);
        b = 0.0;

        // Heat flux on top surface
        ConstantCoefficient heat_flux_coeff(heat_flux);
        Array<int> flux_bdr(pmesh->bdr_attributes.Max());
        flux_bdr = 0;
        flux_bdr[top_attr - 1] = 1;
        b.AddBdrFaceIntegrator(new BoundaryLFIntegrator(heat_flux_coeff), flux_bdr);

        // Convection on curved surface
        ConstantCoefficient h_T_amb_coeff(h_conv * T_ambient);
        b.AddBdrFaceIntegrator(new BoundaryLFIntegrator(h_T_amb_coeff), conv_bdr);
        b.Assemble();

        // d. Form linear system A*U = B
        HypreParMatrix A;
        Vector B, U;
        a.FormLinearSystem(ess_tdof_list, u_gf, b, A, U, B);

        // e. Solve using PCG with AMG preconditioner
        HypreBoomerAMG amg(A);
        HyprePCG pcg(A);
        pcg.SetTol(1e-12);
        pcg.SetMaxIter(200);
        pcg.SetPrintLevel(0);
        pcg.SetPreconditioner(amg);
        pcg.Mult(B, U);

        // f. Update grid function
        a.RecoverFEMSolution(U, b, u_gf);

        // g. Check convergence
        u_change.Set(1.0, *u_gf.GetTrueDofs());
        u_change -= *u_prev;
        double change_norm = u_change.Norml2();
        double u_norm = u_gf.GetTrueDofs()->Norml2();
        *u_prev = *u_gf.GetTrueDofs();

        if (myid == 0)
        {
            cout << "Iteration " << iter << ": |u_new - u_old|/|u_new| = "
                 << scientific << setprecision(4) << change_norm / u_norm << endl;
        }

        if (change_norm / u_norm < tol)
        {
            if (myid == 0)
            {
                cout << "\n✓ Convergence reached after " << iter << " iterations!" << endl;
            }
            break;
        }

        if (iter == max_iter && myid == 0)
        {
            cout << "\n⚠ Maximum iterations reached without full convergence" << endl;
        }
    }

    delete u_prev;

    // ---------------------------------------------------------------
    // --- COMPUTE AND DISPLAY STATISTICS ---
    // ---------------------------------------------------------------

    double T_min = u_gf.Min();
    double T_max = u_gf.Max();

    if (myid == 0)
    {
        cout << "\n"
             << string(60, '=') << endl;
        cout << "Solution Statistics:" << endl;
        cout << string(60, '=') << endl;
        cout << "  Minimum temperature: " << fixed << setprecision(2)
             << T_min << " °C" << endl;
        cout << "  Maximum temperature: " << T_max << " °C" << endl;
        cout << "  Temperature range:   " << (T_max - T_min) << " °C" << endl;
        cout << string(60, '=') << endl;
    }

    // ---------------------------------------------------------------
    // --- SAVE RESULTS ---
    // ---------------------------------------------------------------

    if (myid == 0)
    {
        fs::create_directories(output_dir);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Save MFEM format
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

    // // Save ParaView format
    // {
    //     ParaViewDataCollection paraview_dc("cylinder_solution", pmesh);
    //     paraview_dc.SetPrefixPath(output_dir);
    //     paraview_dc.RegisterField("temperature", &u_gf);
    //     paraview_dc.SetDataFormat(VTKFormat::BINARY);
    //     paraview_dc.SetHighOrderOutput(true);
    //     paraview_dc.SetLevelsOfDetail(order);
    //     paraview_dc.SetOwnData(false);
    //     paraview_dc.Save();
    // }

    if (myid == 0)
    {
        cout << "\n✓ Results saved to: " << output_dir << "/" << endl;
        cout << "  - MFEM format: " << mfem_mesh << ".* and " << steady_sol << ".*" << endl;
        // cout << "  - ParaView format: cylinder_solution.pvtu" << endl;
        cout << "\nTo visualize with GLVis:" << endl;
        cout << "  ./../glvis-4.4/glvis -m " << output_dir << "/" << mfem_mesh << ".000000 -g "
             << output_dir << "/" << steady_sol << ".000000" << endl;
        // cout << "\nTo visualize with ParaView:" << endl;
        // cout << "  paraview " << output_dir << "/cylinder_solution.pvtu" << endl;
    }

    // Clean up
    delete pfespace;
    delete pmesh;

    Mpi::Finalize();
    return 0;
}