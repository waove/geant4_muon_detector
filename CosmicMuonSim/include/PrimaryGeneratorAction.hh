#ifndef PrimaryGeneratorAction_hh
#define PrimaryGeneratorAction_hh

#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4ParticleGun.hh"
#include "globals.hh"

#include <algorithm>
#include <memory>

class G4Event;

class PrimaryGeneratorAction : public G4VUserPrimaryGeneratorAction {
public:
    PrimaryGeneratorAction();
    ~PrimaryGeneratorAction() override;
    void GeneratePrimaries(G4Event* event) override;

private:
    G4ParticleGun* fParticleGun;

    // ── 2D inverse-CDF sampling table ──────────────────────────────────────
    // Grid dimensions — 500×500 gives sub-percent interpolation error
    // across the full (E, cosθ) domain at negligible memory cost (~2 MB).
    static constexpr int kNE   = 500;
    static constexpr int kNCos = 500;

    G4double fEGrid  [kNE];               // log-spaced energy grid [GeV]
    G4double fCosGrid[kNCos];             // linear cosθ grid [0, 1]
    G4double fCdfE   [kNE];               // marginal CDF in E
    G4double fCdfCosGivenE[kNE][kNCos];  // conditional CDF in cosθ | E

    // Build the table once at construction
    void BuildSamplingTable();

    // Draw one (energy [Geant4 units], cosθ) pair from the table
    void SampleFromTable(G4double& energy, G4double& cosTheta) const;

    // Eq. 2 — Earth-curvature correction (Chirkin/Volkova)
    static G4double CosTheta_star(G4double cosTheta);

    // Eq. 3 — Modified Gaisser differential flux [arbitrary units]
    // E_GeV in GeV, cosTheta is the observed zenith cosine
    static G4double Gaisser_Eq3(G4double E_GeV, G4double cosTheta);
};

#endif
