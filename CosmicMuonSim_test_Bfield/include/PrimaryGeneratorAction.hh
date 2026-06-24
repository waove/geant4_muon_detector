#ifndef PrimaryGeneratorAction_hh
#define PrimaryGeneratorAction_hh

#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4ParticleGun.hh"
#include "globals.hh"

class G4Event;

// ==============================================================
// PrimaryGeneratorAction — MAGNETIC FIELD TEST MODE
//
// Replaces the cosmic muon source with a low-energy electron gun
// placed in the air gap between the Al plate and Scint 2, inside
// the copper coil where the B-field (1 T test value) is active.
//
// Electrons are fired along +y (perpendicular to B along +z)
// so that the Lorentz force produces visible spirals in x-y.
//
// Energy range: 0.5 – 3 MeV  (uniform random)
//   → Larmor radius 2.5 – 12 mm at B = 1 T
//   → fits inside the 45 mm air gap
// ==============================================================

class PrimaryGeneratorAction : public G4VUserPrimaryGeneratorAction {
public:
    PrimaryGeneratorAction();
    ~PrimaryGeneratorAction() override;
    void GeneratePrimaries(G4Event* event) override;

private:
    G4ParticleGun* fParticleGun;
};

#endif
