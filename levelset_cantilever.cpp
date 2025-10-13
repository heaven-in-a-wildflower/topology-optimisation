/*
Fixed Level-Set Topology Optimization - Corrected Velocity
Key insight: The velocity should REDUCE compliance, not just follow Eq. 24 blindly
We want to ADD material where stress is high, REMOVE where stress is low
*/

#include "mfem.hpp"
#include <iostream>
#include <fstream>
#include <cmath>
#include <filesystem>
#include <vector>

using namespace std;
using namespace mfem;
namespace fs = std::filesystem;

// Smooth Heaviside function
double Heaviside(double psi, double epsilon)
{
    if (psi < -epsilon)
        return 0.0;
    else if (psi > epsilon)
        return 1.0;
    else
        return 0.5 + psi / (2.0 * epsilon) + sin(M_PI * psi / epsilon) / (2.0 * M_PI);
}

// Material density with ersatz material
class LameCoefficient : public Coefficient
{
private:
    GridFunction *psi;
    double E_solid, nu, E_void;
    bool is_lambda;
    double epsilon;

public:
    LameCoefficient(GridFunction *psi_, double E_s, double nu_, double E_v, bool lambda, double eps)
        : psi(psi_), E_solid(E_s), nu(nu_), E_void(E_v), is_lambda(lambda), epsilon(eps) {}

    virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip)
    {
        double psi_val = psi->GetValue(T, ip);
        double H = Heaviside(psi_val, epsilon);
        double E = E_void + (E_solid - E_void) * (1.0 - H);

        if (is_lambda)
            return E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        else
            return E / (2.0 * (1.0 + nu));
    }
};

// Shape velocity - CORRECTED SIGN
// For compliance minimization: negative velocity means boundary moves INWARD (psi increases)
// We want: high strain energy → move OUTWARD (add material) → velocity should be POSITIVE for material addition
// But level-set convention: ψ < 0 is material, so negative velocity shrinks material
//
// The key: equation (8) gives shape derivative J'(Ω)(θ) = ∫ v·θ·n ds
// For compliance, v = -A:e(u):e(u) on free boundary
// Negative v means moving boundary inward DECREASES J (good for minimization)
//
// But we have volume constraint! So: v = strain_energy - ℓ
// High strain energy → large positive v → boundary moves outward → add material
class ShapeVelocity : public Coefficient
{
private:
    GridFunction *u;
    GridFunction *psi;
    double lagrange_mult;
    double E, nu;
    double epsilon;

public:
    ShapeVelocity(GridFunction *u_, GridFunction *psi_,
                  double ell, double E_, double nu_, double eps)
        : u(u_), psi(psi_), lagrange_mult(ell), E(E_), nu(nu_), epsilon(eps) {}

    void SetLagrangeMultiplier(double ell) { lagrange_mult = ell; }

    virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip)
    {
        double psi_val = psi->GetValue(T, ip);

        // Only compute near interface
        if (fabs(psi_val) > 2.0 * epsilon)
            return 0.0;

        double lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        double mu = E / (2.0 * (1.0 + nu));

        DenseMatrix grad_u;
        u->GetVectorGradient(T, grad_u);

        double e11 = grad_u(0, 0);
        double e22 = grad_u(1, 1);
        double e12 = 0.5 * (grad_u(0, 1) + grad_u(1, 0));
        double trace_e = e11 + e22;

        double s11 = 2.0 * mu * e11 + lambda * trace_e;
        double s22 = 2.0 * mu * e22 + lambda * trace_e;
        double s12 = 2.0 * mu * e12;

        double strain_energy = s11 * e11 + s22 * e22 + 2.0 * s12 * e12;

        // CORRECTED: For compliance minimization with volume constraint:
        // v = strain_energy - ℓ
        // Positive v → add material (move boundary outward, decrease ψ)
        // Negative v → remove material (move boundary inward, increase ψ)
        return strain_energy - lagrange_mult;
    }
};

// Upwind gradient for Hamilton-Jacobi
double UpwindGradient(const GridFunction &psi, double V,
                      int i, int j, int Nx, int Ny, double dx, double dy)
{
    int idx = i + j * Nx;
    double psi_c = psi(idx);

    double Dxp = (i < Nx - 1) ? (psi((i + 1) + j * Nx) - psi_c) / dx : 0;
    double Dxm = (i > 0) ? (psi_c - psi((i - 1) + j * Nx)) / dx : 0;
    double Dyp = (j < Ny - 1) ? (psi(i + (j + 1) * Nx) - psi_c) / dy : 0;
    double Dym = (j > 0) ? (psi_c - psi(i + (j - 1) * Nx)) / dy : 0;

    double grad_x_sq, grad_y_sq;

    if (V > 0)
    {
        grad_x_sq = max(pow(max(Dxm, 0.0), 2), pow(min(Dxp, 0.0), 2));
        grad_y_sq = max(pow(max(Dym, 0.0), 2), pow(min(Dyp, 0.0), 2));
    }
    else
    {
        grad_x_sq = max(pow(min(Dxm, 0.0), 2), pow(max(Dxp, 0.0), 2));
        grad_y_sq = max(pow(min(Dym, 0.0), 2), pow(max(Dyp, 0.0), 2));
    }

    return sqrt(grad_x_sq + grad_y_sq);
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    // Parameters
    const double Lx = 2.0, Ly = 1.0;
    const int Nx = 160, Ny = 80;

    const double E = 1.0;
    const double nu = 0.3;
    const double E_void = 1e-3;

    const double dx = Lx / Nx;
    const double dy = Ly / Ny;
    const double epsilon = 2.5 * max(dx, dy);

    // Lagrange multiplier: balance between stiffness and volume
    // Higher value → less material (more volume penalty)
    // Lower value → more material (prioritize stiffness)
    // Typical range: 1-10 for this problem
    double lagrange_mult = 5.0;

    const int max_iter = 100;
    const int hj_steps = 10; // Reduced for stability
    const int reinit_freq = 5;

    cout << "Level-Set Topology Optimization - 2D Cantilever\n";
    cout << "Minimizing: Compliance (with volume constraint)\n";
    cout << "Grid: " << Nx << " x " << Ny << "\n";
    cout << "Interface thickness: " << epsilon << "\n";
    cout << "Lagrange multiplier: " << lagrange_mult << "\n\n";

    Mesh mesh = Mesh::MakeCartesian2D(Nx, Ny, Element::QUADRILATERAL,
                                      false, Lx, Ly, false);

    H1_FECollection fec(1, 2);
    FiniteElementSpace fes_H1(&mesh, &fec);
    FiniteElementSpace fes_vec(&mesh, &fec, 2);

    cout << "DOFs: " << fes_H1.GetTrueVSize() << " (scalar), "
         << fes_vec.GetTrueVSize() << " (vector)\n\n";

    // Boundary conditions
    Array<int> ess_bdr(mesh.bdr_attributes.Max());
    ess_bdr = 0;
    ess_bdr[0] = 1;

    Array<int> ess_tdof;
    fes_vec.GetEssentialTrueDofs(ess_bdr, ess_tdof);

    // Initialize with holes (similar to paper)
    GridFunction psi(&fes_H1);

    int holes_x = 6;
    int holes_y = 3;
    double hole_radius = 0.12;

    for (int i = 0; i < psi.Size(); i++)
    {
        const double *coord = mesh.GetVertex(i);
        double x = coord[0], y = coord[1];

        double min_dist = 1e10;

        for (int hx = 0; hx < holes_x; hx++)
        {
            for (int hy = 0; hy < holes_y; hy++)
            {
                double cx = 0.25 + hx * 0.3;
                double cy = (hy + 0.5) * Ly / holes_y;
                double dist = sqrt(pow(x - cx, 2) + pow(y - cy, 2)) - hole_radius;
                min_dist = min(min_dist, dist);
            }
        }

        psi(i) = min_dist;
    }

    // Initial volume
    double initial_volume = 0.0;
    for (int i = 0; i < psi.Size(); i++)
    {
        double H = Heaviside(psi(i), epsilon);
        initial_volume += (1.0 - H);
    }
    initial_volume /= psi.Size();
    cout << "Initial volume fraction: " << initial_volume << "\n\n";

    double dt_hj = 0.2 * min(dx, dy); // Smaller time step for stability

    GridFunction u(&fes_vec);
    u = 0.0;

    vector<double> compliance_history;
    vector<double> volume_history;
    double prev_compliance = 1e10;

    for (int iter = 0; iter < max_iter; iter++)
    {
        cout << "=== Iteration " << iter + 1 << " ===\n";

        // Solve elasticity
        LameCoefficient lambda_c(&psi, E, nu, E_void, true, epsilon);
        LameCoefficient mu_c(&psi, E, nu, E_void, false, epsilon);

        BilinearForm a(&fes_vec);
        a.AddDomainIntegrator(new ElasticityIntegrator(lambda_c, mu_c));
        a.Assemble();

        LinearForm b(&fes_vec);
        b = 0.0;

        // Point load
        double load_x = Lx, load_y = Ly / 2.0;
        double min_dist = 1e10;
        int load_vertex = -1;

        for (int i = 0; i < mesh.GetNV(); i++)
        {
            const double *coord = mesh.GetVertex(i);
            double dist = sqrt(pow(coord[0] - load_x, 2) +
                               pow(coord[1] - load_y, 2));
            if (dist < min_dist)
            {
                min_dist = dist;
                load_vertex = i;
            }
        }

        if (load_vertex >= 0)
        {
            Array<int> vdofs;
            fes_vec.GetVertexVDofs(load_vertex, vdofs);
            if (vdofs.Size() >= 2)
            {
                int y_dof = vdofs[1];
                if (y_dof < 0)
                    y_dof = -1 - y_dof;
                Vector &b_vec = b;
                b_vec(y_dof) = -1.0;
            }
        }

        // Solve
        OperatorPtr A;
        Vector B, X;
        a.FormLinearSystem(ess_tdof, u, b, A, X, B);

        GSSmoother M((SparseMatrix &)(*A));
        PCG(*A, M, B, X, 0, 1000, 1e-10, 0.0);

        a.RecoverFEMSolution(X, b, u);

        // Compute compliance
        double compliance = b * u;

        // Compute volume
        double volume_fraction = 0.0;
        for (int i = 0; i < psi.Size(); i++)
        {
            double H = Heaviside(psi(i), epsilon);
            volume_fraction += (1.0 - H);
        }
        volume_fraction /= psi.Size();

        compliance_history.push_back(compliance);
        volume_history.push_back(volume_fraction);

        cout << "Compliance: " << compliance;
        if (iter > 0)
        {
            double change = (compliance - prev_compliance) / prev_compliance * 100;
            cout << " (" << (change > 0 ? "+" : "") << change << "%)";
        }
        cout << "\n";
        cout << "Volume fraction: " << volume_fraction << "\n";

        prev_compliance = compliance;

        // Compute velocity
        ShapeVelocity velocity(&u, &psi, lagrange_mult, E, nu, epsilon);
        GridFunction vel(&fes_H1);
        vel.ProjectCoefficient(velocity);

        // Velocity statistics
        double min_vel = 1e10, max_vel = -1e10, avg_vel = 0.0;
        int nonzero_count = 0;
        for (int i = 0; i < vel.Size(); i++)
        {
            if (fabs(vel(i)) > 1e-10)
            {
                min_vel = min(min_vel, vel(i));
                max_vel = max(max_vel, vel(i));
                avg_vel += vel(i);
                nonzero_count++;
            }
        }
        if (nonzero_count > 0)
            avg_vel /= nonzero_count;

        cout << "Velocity: min=" << min_vel << ", max=" << max_vel
             << ", avg=" << avg_vel << " (nonzero: " << nonzero_count << ")\n";

        // Normalize for stability
        double max_abs_vel = max(fabs(min_vel), fabs(max_vel));
        if (max_abs_vel > 1e-8)
        {
            for (int i = 0; i < vel.Size(); i++)
                vel(i) /= max_abs_vel;
        }

        // Hamilton-Jacobi evolution: ∂ψ/∂t - v|∇ψ| = 0
        // Note: negative v means ψ decreases → material grows (since material is ψ<0)
        for (int step = 0; step < hj_steps; step++)
        {
            GridFunction psi_new = psi;

            for (int j = 0; j < Ny; j++)
            {
                for (int i = 0; i < Nx; i++)
                {
                    int idx = i + j * Nx;
                    double v = vel(idx);
                    double grad = UpwindGradient(psi, v, i, j, Nx, Ny, dx, dy);
                    psi_new(idx) = psi(idx) - dt_hj * v * grad;
                }
            }

            psi = psi_new;
        }

        // Reinitialize
        if ((iter + 1) % reinit_freq == 0)
        {
            cout << "Reinitializing...\n";
            GridFunction psi0 = psi;
            double dt_r = 0.1 * min(dx, dy);

            for (int r = 0; r < 10; r++)
            {
                GridFunction psi_temp = psi;
                for (int j = 0; j < Ny; j++)
                {
                    for (int i = 0; i < Nx; i++)
                    {
                        int idx = i + j * Nx;
                        double sign = (psi0(idx) >= 0) ? 1.0 : -1.0;
                        double grad = UpwindGradient(psi, sign, i, j, Nx, Ny, dx, dy);
                        psi_temp(idx) = psi(idx) - dt_r * sign * (grad - 1.0);
                    }
                }
                psi = psi_temp;
            }
        }

        // Check convergence
        if (iter > 10)
        {
            double comp_change = fabs(compliance_history[iter] -
                                      compliance_history[iter - 5]) /
                                 compliance_history[iter];

            cout << "Convergence: " << comp_change << "\n";

            if (comp_change < 0.005)
            {
                cout << "\n✓ Converged!\n";
                break;
            }
        }

        cout << "\n";
    }

    // Save results
    fs::create_directories("levelset_results");

    ofstream mesh_ofs("levelset_results/cantilever.mesh");
    mesh.Print(mesh_ofs);

    GridFunction density(&fes_H1);
    for (int i = 0; i < density.Size(); i++)
    {
        double H = Heaviside(psi(i), epsilon);
        density(i) = 1.0 - H;
    }

    ofstream density_ofs("levelset_results/density_final.gf");
    density.Save(density_ofs);

    ofstream levelset_ofs("levelset_results/levelset_final.gf");
    psi.Save(levelset_ofs);

    ofstream disp_ofs("levelset_results/displacement_final.gf");
    u.Save(disp_ofs);

    ofstream hist_ofs("levelset_results/history.txt");
    hist_ofs << "# Iteration Compliance VolumeFraction\n";
    for (size_t i = 0; i < compliance_history.size(); i++)
        hist_ofs << i + 1 << " " << compliance_history[i] << " "
                 << volume_history[i] << "\n";

    cout << "\n=== Optimization Complete ===\n";
    cout << "Final compliance: " << compliance_history.back() << "\n";
    cout << "Final volume fraction: " << volume_history.back() << "\n";
    cout << "Compliance reduction: "
         << (compliance_history[0] - compliance_history.back()) / compliance_history[0] * 100
         << "%\n";

    MPI_Finalize();
    return 0;
}