/*
To compile:
mpicxx -O3 -std=c++17 -fopenmp -I../mfem-4.8/ -I../hypre/src/hypre/include simp.cpp -o simp -L../mfem-4.8 -lmfem -L../hypre/src/hypre/lib -lHYPRE -L../metis-4.0 -lmetis -lrt -lopenblas 

To run:
./simp

To visualise:
./../glvis-4.4/glvis -m simp_steady_state_solns/steady_thermal.mesh -g simp_steady_state_solns/steady_thermal_final.gf 
*/
//========================================================================
// Steady-state thermal topology optimization (FIXED VERSION)
#include "mfem.hpp"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>

using namespace std;
using namespace mfem;
namespace fs = std::filesystem;

// ============================================================================
// COEFFICIENT CLASSES
// ============================================================================

// Steady-state heat source (constant)
class SteadyHeatSourceCoeff : public Coefficient
{
private:
    double source_magnitude;

public:
    SteadyHeatSourceCoeff(double magnitude = 1.0)
        : source_magnitude(magnitude) {}

    virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip)
    {
        double coords[3];
        Vector x(coords, T.GetSpaceDim());
        T.Transform(ip, x);

        // Heat source at center
        // if (x(0) < 0.01 && x(1)>0.49 && x(1)<0.51)
        // if (x(0) > 0.495 && x(0) < 0.505 && x(1) > 0.495 && x(1) < 0.505)
        // if (x(0) < 0.01 || x(0) > 0.96 || x(1) < 0.01 || x(1) > 0.96)
        // if ((x(0) > 0 && x(0) < 0.01 && x(1) > 0 && x(1) < 0.01) ||
        //     (x(0) > 0.99 && x(0) < 1 && x(1) > 0 && x(1) < 0.01) ||
        //     (x(0) > 0.99 && x(0) < 1 && x(1) > 0.99 && x(1) < 1) ||
        //     (x(0) > 0 && x(0) < 0.01 && x(1) > 0.99 && x(1) < 1))
        if (x(0) > 0 && x(0) < 1 && x(1) > 0 && x(1) < 1)
            return source_magnitude;
        else
            return 0.0;
    }
};

// Convection coefficient - FULL BOUNDARY
class LocalConvectionCoeff : public Coefficient
{
public:
    double hval;
    LocalConvectionCoeff(double h) : hval(h) {}

    virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip)
    {
        Vector x;
        T.Transform(ip, x);

        // Convection on ALL boundaries (option 1: natural BC approach)
        // This ensures sufficient heat dissipation
        // if (x(0) < 0.01 || x(0) > 0.96 || x(1) < 0.01 || x(1) > 0.96)
        // if ((x(0) > 0 && x(0) < 0.01 && x(1) > 0 && x(1) < 0.01) ||
        //     (x(0) > 0.96 && x(0) < 1 && x(1) > 0 && x(1) < 0.01) ||
        //     (x(0) > 0.96 && x(0) < 1 && x(1) > 0.96 && x(1) < 1) ||
        //     (x(0) > 0 && x(0) < 0.01 && x(1) > 0.96 && x(1) < 1))
        // if(x(0)>0.99)
        if (x(0)<0.01 && x(1)>0.45 && x(1)<0.55)
            return hval;
        else
            return 0.0;
    }
};

// h * T∞ coefficient for convection BC
class LocalConvectionAmbientCoeff : public Coefficient
{
public:
    double hval, Tamb;
    LocalConvectionAmbientCoeff(double h, double T) : hval(h), Tamb(T) {}

    virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip)
    {
        Vector x;
        T.Transform(ip, x);

        // if ((x(0) > 0 && x(0) < 0.01 && x(1) > 0 && x(1) < 0.01) ||
        //     (x(0) > 0.96 && x(0) < 1 && x(1) > 0 && x(1) < 0.01) ||
        //     (x(0) > 0.96 && x(0) < 1 && x(1) > 0.96 && x(1) < 1) ||
        //     (x(0) > 0 && x(0) < 0.01 && x(1) > 0.96 && x(1) < 1))
        // if(x(0)>0.99)
        if (x(0)<0.01 && x(1)>0.45 && x(1)<0.55)
            return hval * Tamb;
        else
            return 0.0;
    }
};

// SIMP thermal conductivity
class ThermalConductivityCoeff : public Coefficient
{
private:
    GridFunction *rho;
    double simp_exponent;
    double rho_min;

public:
    ThermalConductivityCoeff(GridFunction *rho_, double simp_exp, double rho_min_val)
        : rho(rho_), simp_exponent(simp_exp), rho_min(rho_min_val) {}

    void SetSIMPExponent(double exp) { simp_exponent = exp; }

    virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip)
    {
        double density = rho->GetValue(T, ip);
        density = max(rho_min, min(1.0, density));

        // k = k_min + (k_max - k_min) * ρ^p
        double k_min = rho_min * 1e-3;
        double k_max = 1.0;

        return k_min + (k_max - k_min) * pow(density, simp_exponent);
    }
};

// ============================================================================
// DENSITY FILTER
// ============================================================================

void densityFilterPDE(Mesh &mesh, GridFunction &rho, GridFunction &rho_filtered,
                      double r_min, Array<int> *ess_bdr = nullptr)
{
    FiniteElementSpace *fes = rho.FESpace();

    BilinearForm filter_form(fes);
    ConstantCoefficient r_squared(r_min * r_min);
    ConstantCoefficient one(1.0);

    filter_form.AddDomainIntegrator(new DiffusionIntegrator(r_squared));
    filter_form.AddDomainIntegrator(new MassIntegrator(one));
    filter_form.Assemble();
    filter_form.Finalize();

    BilinearForm mass_form(fes);
    mass_form.AddDomainIntegrator(new MassIntegrator(one));
    mass_form.Assemble();
    mass_form.Finalize();

    Vector rhs(fes->GetNDofs());
    mass_form.Mult(rho, rhs);

    Array<int> empty_ess_tdof;
    if (ess_bdr == nullptr)
    {
        ess_bdr = &empty_ess_tdof;
    }

    SparseMatrix &A = filter_form.SpMat();

    CGSolver cg;
    cg.SetMaxIter(1000);
    cg.SetRelTol(1e-12);
    cg.SetAbsTol(1e-15);
    cg.SetPrintLevel(0);
    cg.SetOperator(A);

    rho_filtered = 0.0;
    cg.Mult(rhs, rho_filtered);

    if (!cg.GetConverged())
    {
        std::cerr << "Warning: Filter did not converge!" << std::endl;
    }
}

// ============================================================================
// STEADY-STATE SENSITIVITY CALCULATION
// ============================================================================

void ComputeSteadyStateSensitivities(
    const GridFunction &temperature,
    const GridFunction &rho,
    GridFunction &sensitivity,
    double simp_exponent,
    double rho_min,
    FiniteElementSpace *fes)
{
    Mesh *mesh = fes->GetMesh();
    int ne = mesh->GetNE();
    sensitivity = 0.0;

    Vector elem_sens, elem_rho;
    DenseMatrix dshape;
    Vector grad_T;

    // Loop over elements
    for (int e = 0; e < ne; e++)
    {
        // Get elem data
        const FiniteElement *fe = fes->GetFE(e);
        ElementTransformation *T = mesh->GetElementTransformation(e);
        int dof = fe->GetDof();
        int dim = fe->GetDim();

        // Get global indices of dofs associated with the elem
        Array<int> vdofs;
        fes->GetElementVDofs(e, vdofs);

        // Extract nodal values of rho and temp
        elem_rho.SetSize(dof);
        Vector elem_temp(dof);
        elem_sens.SetSize(dof);
        rho.GetSubVector(vdofs, elem_rho);
        temperature.GetSubVector(vdofs, elem_temp);
        elem_sens = 0.0;

        // set up quadrature
        // choose an integration rule of order 2×element order to get sufficient accuracy.
        const IntegrationRule *ir = &IntRules.Get(fe->GetGeomType(), 2 * fe->GetOrder());
        dshape.SetSize(dof, dim);
        grad_T.SetSize(dim);

        for (int i = 0; i < ir->GetNPoints(); i++)
        {
            // map from ref elem to physical elem
            const IntegrationPoint &ip = ir->IntPoint(i);
            T->SetIntPoint(&ip);
            double w = T->Weight() * ip.weight;

            Vector shape(dof);
            fe->CalcShape(ip, shape); // value of shape function at that point
            fe->CalcDShape(ip, dshape); // derivative of shape function in ref coords

            // Compute interpolated density
            double rho_ip = 0.0;
            for (int j = 0; j < dof; j++)
                rho_ip += elem_rho(j) * shape(j);
            rho_ip = max(rho_min, min(1.0, rho_ip));

            // Compute delta T in ref and phys coords
            Vector grad_T_ref(dim);
            grad_T_ref = 0.0;
            for (int j = 0; j < dof; j++)
                for (int d = 0; d < dim; d++)
                    grad_T_ref(d) += elem_temp(j) * dshape(j, d);

            T->InverseJacobian().MultTranspose(grad_T_ref, grad_T);

            // compute |delta T|**2
            double grad_T_squared = 0.0;
            for (int d = 0; d < dim; d++)
                grad_T_squared += grad_T(d) * grad_T(d);

            // compute dk_drho
            double k_min = rho_min * 1e-3;
            double k_max = 1.0;
            double dk_drho = (k_max - k_min) * simp_exponent * pow(rho_ip, simp_exponent - 1.0);

            // compute local sensitivity contribution : ∂ρ/∂J​=−(∂ρ/∂k)*​∣∇T∣**2
            double local_sensitivity = -dk_drho * grad_T_squared;

            // accumulate weighted density
            for (int j = 0; j < dof; j++)
                elem_sens(j) += local_sensitivity * shape(j) * w;
        }

        // add elem contribution to global sensitivity
        sensitivity.AddElementVector(vdofs, elem_sens);
    }
}

// ============================================================================
// MAIN PROGRAM
// ============================================================================

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    // ========================================================================
    // PARAMETERS
    // ========================================================================

    // Choose boundary condition approach:
    // Option 1: Use convection on full boundary (RECOMMENDED for steady-state)
    // Option 2: Use fixed temperature on one edge (classical approach)
    const bool USE_CONVECTION_BC = true;  // Set to false for Dirichlet BC

    // Optimization parameters
    const double vol_frac = 0.3;
    const double rho_min = 1e-6;
    const int max_iter = 150;
    const int non_penalized_iter = 10;
    const double max_simp_exp = 8.0;
    const double tolerance = 1e-5;
    const double move_limit = 0.05;
    const double r_min = 0.02;

    // Boundary conditions
    const double h_val = 50.0;           // Increased convection coefficient
    const double T_amb = 25.0;           // Ambient temperature (°C)
    const double T_fixed = 25.0;         // Fixed temperature for Dirichlet BC
    const double Q_magnitude = 1.0;   // Heat source magnitude (W/m²)

    cout << "==================================================" << endl;
    cout << "STEADY-STATE THERMAL TOPOLOGY OPTIMIZATION" << endl;
    cout << "==================================================" << endl;
    cout << "BC Type: " << (USE_CONVECTION_BC ? "Convection (Natural)" : "Fixed Temperature (Dirichlet)") << endl;
    cout << "Target volume fraction: " << vol_frac << endl;
    cout << "Heat source magnitude: " << Q_magnitude << " W/m²" << endl;
    if (USE_CONVECTION_BC) {
        cout << "Convection coefficient: " << h_val << " W/(m²·K)" << endl;
        cout << "Ambient temperature: " << T_amb << " °C" << endl;
    } else {
        cout << "Fixed boundary temperature: " << T_fixed << " °C" << endl;
    }
    cout << "==================================================" << endl;

    // ========================================================================
    // MESH AND FINITE ELEMENT SPACE
    // ========================================================================

    Mesh mesh = Mesh::MakeCartesian2D(80, 80, Element::QUADRILATERAL,
                                      false, 1.0, 1.0, false);

    H1_FECollection fec(1, mesh.Dimension());
    FiniteElementSpace fes(&mesh, &fec);

    cout << "Number of unknowns: " << fes.GetNDofs() << endl;

    // ========================================================================
    // BOUNDARY CONDITIONS
    // ========================================================================

    Array<int> ess_tdof_list;
    
    if (!USE_CONVECTION_BC)
    {
        // Option 2: Fixed temperature on left boundary
        double tol = 1e-6;
        for (int i = 0; i < fes.GetNBE(); i++)
        {
            Array<int> vdofs;
            fes.GetBdrElementVDofs(i, vdofs);

            for (int j = 0; j < vdofs.Size(); j++)
            {
                int dof = vdofs[j];
                if (dof < 0) dof = -1 - dof;

                int vertex_id = dof;
                const double *x = mesh.GetVertex(vertex_id);

                // Fix temperature on left edge
                if (x[0] < 0.01 + tol)
                {
                    if (ess_tdof_list.Find(dof) < 0)
                        ess_tdof_list.Append(dof);
                }
            }
        }
        cout << "Fixed temperature DOFs: " << ess_tdof_list.Size() << endl;
    }
    else
    {
        cout << "Using convection boundary conditions (no essential BCs)" << endl;
    }

    // ========================================================================
    // INITIALIZE FIELDS
    // ========================================================================

    GridFunction rho(&fes), rho_filtered(&fes), sensitivity(&fes);
    rho = vol_frac;
    sensitivity = 0.0;

    // ========================================================================
    // SETUP COEFFICIENTS
    // ========================================================================

    SteadyHeatSourceCoeff source_coeff(Q_magnitude);
    LocalConvectionCoeff h_coeff(h_val);
    LocalConvectionAmbientCoeff hTinf_coeff(h_val, T_amb);

    // ========================================================================
    // OPTIMIZATION LOOP
    // ========================================================================

    double simp_exponent = 1.0;
    int exp_update_counter = 0;
    vector<double> compliance_history;
    double old_compliance = 1e30;

    CGSolver cg_solver;
    cg_solver.SetMaxIter(20000);
    cg_solver.SetRelTol(1e-6);
    cg_solver.SetAbsTol(1e-9);
    cg_solver.SetPrintLevel(-1);  // Suppress warnings initially

    for (int iter = 0; iter < max_iter; iter++)
    {
        cout << "\n================================================" << endl;
        cout << "Iteration " << iter + 1 << "/" << max_iter;
        cout << " (SIMP p=" << simp_exponent << ")" << endl;
        cout << "================================================" << endl;

        // ====================================================================
        // STEP 1: FILTER DESIGN VARIABLES
        // ====================================================================
        densityFilterPDE(mesh, rho, rho_filtered, r_min);

        // ====================================================================
        // STEP 2: STEADY-STATE FORWARD ANALYSIS
        // ====================================================================

        ThermalConductivityCoeff k_coeff(&rho_filtered, simp_exponent, rho_min);

        GridFunction temperature(&fes);
        temperature = T_amb;

        // Set fixed boundary temperatures if using Dirichlet BC
        if (!USE_CONVECTION_BC)
        {
            for (int i = 0; i < ess_tdof_list.Size(); i++)
            {
                temperature[ess_tdof_list[i]] = T_fixed;
            }
        }

        cout << "Running steady-state analysis..." << endl;

        // Stiffness matrix
        BilinearForm k_form(&fes);
        k_form.AddDomainIntegrator(new DiffusionIntegrator(k_coeff));
        
        if (USE_CONVECTION_BC)
        {
            // Add convection on boundaries
            k_form.AddBoundaryIntegrator(new BoundaryMassIntegrator(h_coeff));
        }
        
        k_form.Assemble();
        k_form.Finalize();

        // Right-hand side
        LinearForm q_form(&fes);
        q_form.AddDomainIntegrator(new DomainLFIntegrator(source_coeff));
        
        if (USE_CONVECTION_BC)
        {
            q_form.AddBoundaryIntegrator(new BoundaryLFIntegrator(hTinf_coeff));
        }
        
        q_form.Assemble();

        // Form linear system
        OperatorPtr A_bc;
        Vector B, X;

        k_form.FormLinearSystem(ess_tdof_list, temperature, q_form, A_bc, X, B);

        // Solve
        cg_solver.SetOperator(*A_bc);
        X = 0.0;
        cg_solver.Mult(B, X);

        if (!cg_solver.GetConverged())
        {
            cout << "WARNING: CG did not converge! Iterations: " << cg_solver.GetNumIterations() << endl;
        }

        // Recover solution
        k_form.RecoverFEMSolution(X, q_form, temperature);

        // Calculate compliance
        double compliance_steady = q_form * temperature;
        compliance_history.push_back(compliance_steady);

        // Volume calculation
        double current_vol = 0.0;
        for (int i = 0; i < rho_filtered.Size(); i++)
            current_vol += rho_filtered(i);
        current_vol /= rho_filtered.Size();

        // Temperature statistics
        double max_temp = temperature.Max();
        double min_temp = temperature.Min();
        double avg_temp = 0.0;
        for (int i = 0; i < temperature.Size(); i++)
            avg_temp += temperature(i);
        avg_temp /= temperature.Size();

        cout << "\nSteady-state compliance: " << compliance_steady << endl;
        cout << "Temperature - Min: " << min_temp << ", Max: " << max_temp 
             << ", Avg: " << avg_temp << " °C" << endl;
        cout << "Volume fraction: " << current_vol << endl;

        // ====================================================================
        // STEP 3: SENSITIVITY ANALYSIS
        // ====================================================================

        GridFunction sensitivity_physical(&fes);
        ComputeSteadyStateSensitivities(temperature, rho_filtered,
                                        sensitivity_physical, simp_exponent,
                                        rho_min, &fes);

        // Chain rule: filter adjoint
        densityFilterPDE(mesh, sensitivity_physical, sensitivity, r_min);

        double min_sens = 1e30, max_sens = -1e30;
        for (int i = 0; i < sensitivity.Size(); i++)
        {
            min_sens = min(min_sens, sensitivity(i));
            max_sens = max(max_sens, sensitivity(i));
        }
        cout << "Sensitivity range: [" << min_sens << ", " << max_sens << "]" << endl;

        // ====================================================================
        // STEP 4: OPTIMALITY CRITERIA UPDATE
        // ====================================================================
        // updates rho to reduce compliance while trying to maintaining volume constraint.
        
        /*
        Save the current densities in rho_old. The OC update uses the previous densities to compute candidate new densities; we don’t want on-the-fly updates influencing other entries.
        */

        // use biscetion method on lagrange multiplier.
        Vector rho_old = rho;
        double lam_min = 1e-10, lam_max = 1e10;

        for (int lam_iter = 0; lam_iter < 50; lam_iter++)
        {
            double lam_mid = 0.5 * (lam_min + lam_max);
            double vol_sum = 0.0;

            // Going over all elems in the mesh
            for (int i = 0; i < rho.Size(); i++)
            {
                double rho_old_i = rho_old(i);
                double sens_i = sensitivity(i);
                double Be = -sens_i / lam_mid;

                double rho_new_i;
                if (Be <= 0)
                {
                    // ideally this should not happen
                    rho_new_i = max(rho_min, rho_old_i - move_limit);
                }
                else
                {
                    // update rho as per Be
                    rho_new_i = max(rho_min, min(1.0, rho_old_i * sqrt(Be)));
                    
                    // apply move constraints
                    rho_new_i = max(rho_old_i - move_limit, rho_new_i);
                    rho_new_i = min(rho_old_i + move_limit, rho_new_i);
                }

                // clamp again
                rho_new_i = max(rho_min, min(1.0, rho_new_i));

                // accumulate predcited total material
                vol_sum += rho_new_i;
            }

            double vol_current = vol_sum / rho.Size();

            if (abs(vol_current - vol_frac) < 1e-6)
                break;

            if (vol_current > vol_frac)
                lam_min = lam_mid;
            else
                lam_max = lam_mid;
        }

        // Apply final update

        // lambda as mid point of shrunk bracket
        double lam_final = 0.5 * (lam_min + lam_max);
        double max_change = 0.0;

        for (int i = 0; i < rho.Size(); i++)
        {
            double rho_old_i = rho_old(i);
            double sens_i = sensitivity(i);
            double Be = -sens_i / lam_final;

            double rho_new_i;
            if (Be <= 0)
            {
                rho_new_i = max(rho_min, rho_old_i - move_limit);
            }
            else
            {
                rho_new_i = max(rho_min, min(1.0, rho_old_i * sqrt(Be)));
                rho_new_i = max(rho_old_i - move_limit, rho_new_i);
                rho_new_i = min(rho_old_i + move_limit, rho_new_i);
            }

            rho_new_i = max(rho_min, min(1.0, rho_new_i));
            //track the largest abs change for convergence monitoring
            max_change = max(max_change, abs(rho_new_i - rho_old_i));
            rho(i) = rho_new_i;
        }

        cout << "Max density change: " << max_change << endl;

        // ====================================================================
        // STEP 5: CONTINUATION
        // ====================================================================

        exp_update_counter++;

        if (iter >= non_penalized_iter)
        {
            if (iter == non_penalized_iter)
                cout << "\n*** Starting penalization ***" << endl;

            if (exp_update_counter >= 5 && simp_exponent < max_simp_exp)
            {
                simp_exponent = min(simp_exponent * 1.5, max_simp_exp);
                exp_update_counter = 0;
                cout << "⚡ SIMP exponent increased to: " << simp_exponent << endl;
            }
        }

        // ====================================================================
        // STEP 6: CONVERGENCE CHECK
        // ====================================================================

        int binary_count = 0;
        for (int i = 0; i < rho_filtered.Size(); i++)
        {
            double rho_val = rho_filtered(i);
            if (rho_val < 0.05 || rho_val > 0.95)
                binary_count++;
        }
        double binary_percent = (double)binary_count / rho_filtered.Size() * 100.0;
        cout << "Binary design: " << binary_percent << "%" << endl;

        if (iter > 15 && abs(compliance_steady - old_compliance) < tolerance * abs(old_compliance))
        {
            cout << "\n✓ Converged after " << iter + 1 << " iterations!" << endl;
            break;
        }

        old_compliance = compliance_steady;

        // ====================================================================
        // STEP 7: OUTPUT
        // ====================================================================

        // if ((iter + 1) % 25 == 0 || iter == 0)
        // {
        //     string filename = "steady_thermal_" + to_string(iter + 1) + ".gf";
        //     ofstream sol_ofs(filename.c_str());
        //     rho_filtered.Save(sol_ofs);
        // }
    }

    // ========================================================================
    // SAVE FINAL RESULTS
    // ========================================================================

    string result_dir = "simp_steady_state_solns/";
    fs::create_directories(result_dir);

    ofstream mesh_ofs(result_dir + "steady_thermal.mesh");
    mesh.Print(mesh_ofs);

    ofstream final_rho(result_dir + "steady_thermal_final.gf");
    rho_filtered.Save(final_rho);

    GridFunction rho_binary(&fes);
    for (int i = 0; i < rho.Size(); i++)
        rho_binary(i) = (rho_filtered(i) > 0.5) ? 1.0 : rho_min;

    ofstream binary_rho(result_dir + "steady_thermal_binary.gf");
    rho_binary.Save(binary_rho);

    cout << "\n==================================================" << endl;
    cout << "OPTIMIZATION COMPLETE" << endl;
    cout << "==================================================" << endl;
    if (compliance_history.size() > 1) {
        cout << "Compliance improvement: "
             << (compliance_history[0] - compliance_history.back()) / compliance_history[0] * 100.0
             << "%" << endl;
    }
    cout << "Final SIMP exponent: " << simp_exponent << endl;
    MPI_Finalize();
    return 0;
}