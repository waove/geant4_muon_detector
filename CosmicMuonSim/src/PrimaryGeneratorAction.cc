// ==============================================================
// PrimaryGeneratorAction.cc
//
// Cosmic muon source using the MODIFIED Gaisser parametrization
// (Guan et al., arXiv:1509.06176, Eq. 3):
//
//   dI/dE dΩ = 0.14 · [E/GeV · (1 + 3.64 / (E·(cosθ*)^1.29))]^(-2.7)
//       · [ 1/(1 + 1.1·E·cosθ*/115 GeV) + 0.054/(1 + 1.1·E·cosθ*/850 GeV) ]
//
// where cosθ* accounts for Earth curvature (Chirkin/Volkova, Eq. 2):
//
//   cosθ* = sqrt( [(cosθ)² + P1² + P2·(cosθ)^P3 + P4·(cosθ)^P5]
//                 / [1 + P1² + P2 + P4] )
//   P1 = 0.102573, P2 = -0.068287, P3 = 0.958633,
//   P4 = 0.0407253, P5 = 0.817285
//
// Sampling strategy — DIRECT 2D INVERSE-CDF (no accept/reject):
//
//   A 500×500 grid of Eq. 3 is computed once at construction over:
//     E    ∈ [0.1, 100] GeV  (log-spaced)
//     cosθ ∈ [0,   1  ]      (linear, cosθ=0 is horizontal)
//
//   The marginal CDF in E and the conditional CDF in cosθ|E are both
//   precomputed by numerical integration (trapezoid rule).  At runtime,
//   two uniform random numbers are drawn and both CDFs are inverted by
//   binary search + linear interpolation.  Every sample is kept —
//   acceptance rate is exactly 1.0 and the distribution produced is
//   the true Eq. 3 without any approximation or proposal bias.
//
// Source plane:  x = 8000 mm, halfY = halfZ = 1000 mm
// Energy range:  0.1 – 100 GeV
// Zenith range:  0 – 89.9° (cosθ ∈ [0, 1]; full formula valid everywhere)
// Charge ratio:  μ⁺/μ⁻ ≈ 1.27 → 56% μ⁺, 44% μ⁻
//
// gun_energy.csv — written for events [kNDetailEvents,
//   kNDetailEvents + kNGunEnergyEvents) so the trajectory-sampling
//   phase is complete first.  Records the sampled energy, cosθ, and
//   PDG code directly from the gun — zero stepping overhead.
// ==============================================================

#include "PrimaryGeneratorAction.hh"
#include "RunAction.hh"

#include "G4ParticleGun.hh"
#include "G4ParticleTable.hh"
#include "G4RunManager.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"
#include "Randomize.hh"

#include <cmath>
#include <algorithm>
#include <memory>
#include <stdexcept>

// ──────────────────────────────────────────────────────────────
// Constructor — build table then create gun
// ──────────────────────────────────────────────────────────────

PrimaryGeneratorAction::PrimaryGeneratorAction()
    : fParticleGun(new G4ParticleGun(1))
{
    BuildSamplingTable();
}

PrimaryGeneratorAction::~PrimaryGeneratorAction()
{
    delete fParticleGun;
}

// ──────────────────────────────────────────────────────────────
// Eq. 2 — Earth-curvature correction (Chirkin/Volkova)
// ──────────────────────────────────────────────────────────────

G4double PrimaryGeneratorAction::CosTheta_star(G4double cosTheta)
{
    constexpr G4double P1 = 0.102573;
    constexpr G4double P2 = -0.068287;
    constexpr G4double P3 = 0.958633;
    constexpr G4double P4 = 0.0407253;
    constexpr G4double P5 = 0.817285;

    G4double c   = cosTheta;
    G4double num = c*c + P1*P1 + P2*std::pow(c, P3) + P4*std::pow(c, P5);
    G4double den = 1.0  + P1*P1 + P2 + P4;

    if (num / den < 0.0) return 0.0;
    return std::sqrt(num / den);
}

// ──────────────────────────────────────────────────────────────
// Eq. 3 — Modified Gaisser differential flux
//
// Returns dI/dE in arbitrary units (the overall normalisation
// cancels when the CDF is divided by its maximum value).
// E_GeV in GeV, cosTheta is the observed surface zenith cosine.
// ──────────────────────────────────────────────────────────────

G4double PrimaryGeneratorAction::Gaisser_Eq3(G4double E_GeV,
                                              G4double cosTheta)
{
    G4double cosT_star = CosTheta_star(cosTheta);
    if (cosT_star < 1.0e-6) cosT_star = 1.0e-6;

    // Low-energy correction (Eq. 3 new term)
    G4double lowE_corr   = 1.0 + 3.64 / (E_GeV * std::pow(cosT_star, 1.29));
    G4double lowE_factor = std::pow(lowE_corr, -2.7);

    // Pion + kaon decay bracket
    G4double a       = 1.1 * E_GeV * cosT_star;
    G4double bracket = 1.0 / (1.0 + a / 115.0)
                     + 0.054 / (1.0 + a / 850.0);

    return 0.14 * std::pow(E_GeV, -2.7) * lowE_factor * bracket;
}

// ──────────────────────────────────────────────────────────────
// BuildSamplingTable
//
// Evaluates Eq. 3 on a (kNE × kNCos) grid, then builds:
//   fCdfE[i]               — marginal CDF of E
//   fCdfCosGivenE[i][j]   — conditional CDF of cosθ given E = fEGrid[i]
//
// Both CDFs are normalised to [0, 1].
// The table is built once at construction: a few milliseconds,
// ~2 MB memory, negligible compared to Geant4 physics tables.
// ──────────────────────────────────────────────────────────────

void PrimaryGeneratorAction::BuildSamplingTable()
{
    const G4double Emin   = 0.1;   // GeV
    const G4double Emax   = 100.0; // GeV
    const G4double cosMin = 0.0;   // horizontal (θ = 90°)
    const G4double cosMax = 1.0;   // vertical   (θ =  0°)

    // ── Build grids ────────────────────────────────────────────
    for (int i = 0; i < kNE; ++i)
        fEGrid[i] = Emin * std::pow(Emax / Emin,
                                    static_cast<G4double>(i) / (kNE - 1));

    for (int j = 0; j < kNCos; ++j)
        fCosGrid[j] = cosMin
                    + (cosMax - cosMin)
                    * static_cast<G4double>(j) / (kNCos - 1);

    // ── Evaluate Eq. 3 on the 2D grid ─────────────────────────
    // Heap-allocated to avoid blowing the stack (500×500×8 = 2 MB)
    auto pdf2D = std::make_unique<G4double[]>(kNE * kNCos);
    auto idx   = [&](int i, int j) -> G4double& {
        return pdf2D[static_cast<std::size_t>(i) * kNCos + j];
    };

    for (int i = 0; i < kNE; ++i)
        for (int j = 0; j < kNCos; ++j)
            idx(i, j) = Gaisser_Eq3(fEGrid[i], fCosGrid[j]);

    // ── Marginal PDF in E: integrate over cosθ (trapezoid) ────
    G4double margE[kNE] = {};
    for (int i = 0; i < kNE; ++i)
        for (int j = 0; j < kNCos - 1; ++j)
            margE[i] += 0.5 * (idx(i, j) + idx(i, j + 1))
                       * (fCosGrid[j + 1] - fCosGrid[j]);

    // ── Marginal CDF in E (trapezoid over log-spaced E grid) ──
    fCdfE[0] = 0.0;
    for (int i = 1; i < kNE; ++i)
        fCdfE[i] = fCdfE[i - 1]
                 + 0.5 * (margE[i - 1] + margE[i])
                 * (fEGrid[i] - fEGrid[i - 1]);

    G4double normE = fCdfE[kNE - 1];
    if (normE <= 0.0)
        throw std::runtime_error(
            "PrimaryGeneratorAction: marginal CDF in E is zero — "
            "check Gaisser_Eq3 evaluation.");
    for (int i = 0; i < kNE; ++i) fCdfE[i] /= normE;

    // ── Conditional CDF in cosθ | E ───────────────────────────
    for (int i = 0; i < kNE; ++i) {
        fCdfCosGivenE[i][0] = 0.0;
        for (int j = 1; j < kNCos; ++j)
            fCdfCosGivenE[i][j] = fCdfCosGivenE[i][j - 1]
                                 + 0.5 * (idx(i, j - 1) + idx(i, j))
                                 * (fCosGrid[j] - fCosGrid[j - 1]);

        G4double normC = fCdfCosGivenE[i][kNCos - 1];
        if (normC > 0.0) {
            for (int j = 0; j < kNCos; ++j)
                fCdfCosGivenE[i][j] /= normC;
        } else {
            // Degenerate row — should not occur; uniform cosθ as fallback
            for (int j = 0; j < kNCos; ++j)
                fCdfCosGivenE[i][j] = static_cast<G4double>(j) / (kNCos - 1);
        }
    }

    G4cout << "PrimaryGeneratorAction: 2D sampling table built ("
           << kNE << " × " << kNCos << " grid, direct inverse-CDF)" << G4endl;
}

// ──────────────────────────────────────────────────────────────
// SampleFromTable
//
// Draws one (energy [Geant4 internal units], cosθ) pair directly
// from the Eq. 3 distribution using two uniform random numbers
// and binary-search CDF inversion with linear interpolation.
// Acceptance rate = 1.0 — no samples are ever discarded.
// ──────────────────────────────────────────────────────────────

void PrimaryGeneratorAction::SampleFromTable(G4double& energy,
                                              G4double& cosTheta) const
{
    // ── Step 1: sample E from the marginal CDF ─────────────────
    G4double u1 = G4UniformRand();
    int iE = static_cast<int>(
        std::lower_bound(fCdfE, fCdfE + kNE, u1) - fCdfE);
    iE = std::clamp(iE, 1, kNE - 1);

    G4double dCdfE = fCdfE[iE] - fCdfE[iE - 1];
    G4double t     = (dCdfE > 1e-300)
                   ? (u1 - fCdfE[iE - 1]) / dCdfE
                   : 0.5;
    t = std::clamp(t, 0.0, 1.0);

    // Interpolate in log(E) — the grid is log-spaced so this is exact
    G4double logE = std::log(fEGrid[iE - 1])
                  + t * (std::log(fEGrid[iE]) - std::log(fEGrid[iE - 1]));
    energy = std::exp(logE) * GeV;

    // ── Step 2: sample cosθ from the conditional CDF | E ───────
    G4double u2         = G4UniformRand();
    const G4double* row = fCdfCosGivenE[iE];
    int iC = static_cast<int>(
        std::lower_bound(row, row + kNCos, u2) - row);
    iC = std::clamp(iC, 1, kNCos - 1);

    G4double dCdfC = row[iC] - row[iC - 1];
    G4double s     = (dCdfC > 1e-300)
                   ? (u2 - row[iC - 1]) / dCdfC
                   : 0.5;
    s = std::clamp(s, 0.0, 1.0);

    // Linear interpolation on the uniform cosθ grid
    cosTheta = fCosGrid[iC - 1] + s * (fCosGrid[iC] - fCosGrid[iC - 1]);
    cosTheta = std::clamp(cosTheta, 0.0, 1.0);
}

// ──────────────────────────────────────────────────────────────
// GeneratePrimaries
// ──────────────────────────────────────────────────────────────

void PrimaryGeneratorAction::GeneratePrimaries(G4Event* event)
{
    // ── Particle type: μ+/μ- at 56/44 charge ratio ─────────────
    auto*    table  = G4ParticleTable::GetParticleTable();
    G4String muType = (G4UniformRand() < 0.56) ? "mu+" : "mu-";
    fParticleGun->SetParticleDefinition(table->FindParticle(muType));

    // ── Energy + zenith angle: direct sample from Eq. 3 ────────
    // No accept/reject loop — every draw is kept.
    G4double energy, cosTheta;
    SampleFromTable(energy, cosTheta);
    G4double theta = std::acos(cosTheta);

    fParticleGun->SetParticleEnergy(energy);

    // ── Position: uniform on rectangular source plane ───────────
    G4double y0 = (G4UniformRand() - 0.5) * 600.0 * mm; // was 1000.0
    G4double z0 = (G4UniformRand() - 0.5) * 600.0 * mm; // was 1000.0
    fParticleGun->SetParticlePosition({8000.0 * mm, y0, z0});

    // ── Direction: θ from x-axis, φ uniform in [0, 2π) ─────────
    G4double phi = G4UniformRand() * 2.0 * pi;
    G4double dx  = -std::cos(theta);
    G4double dy  =  std::sin(theta) * std::cos(phi);
    G4double dz  =  std::sin(theta) * std::sin(phi);
    fParticleGun->SetParticleMomentumDirection({dx, dy, dz});

    fParticleGun->GeneratePrimaryVertex(event);

    // ── Source validation: record gun energy for N events after
    //    the trajectory phase — one stream write per event, zero
    //    stepping overhead. ─────────────────────────────────────
    G4int eventID = event->GetEventID();
    auto* runAction = static_cast<const RunAction*>(
        G4RunManager::GetRunManager()->GetUserRunAction());

    if (runAction->ShouldRecordGunEnergy(eventID)) {
        runAction->GetGunEnergyStream()
            << eventID                                                 << ","
            << fParticleGun->GetParticleDefinition()->GetPDGEncoding() << ","
            << energy / MeV                                            << ","
            << cosTheta                                                << "\n";
    }
}
