// ==============================================================
// ScintHit.hh — Data object for one recorded interaction
//
// Uses standard new/delete instead of G4Allocator to avoid
// issues with G4String members in MT task-based scheduling.
// ==============================================================

#ifndef SCINT_HIT_HH
#define SCINT_HIT_HH

#include "G4VHit.hh"
#include "G4THitsCollection.hh"
#include "G4ThreeVector.hh"
#include "G4Types.hh"

#include <string>

class ScintHit : public G4VHit
{
public:
    ScintHit() = default;
    ~ScintHit() override = default;

    // Use standard new/delete — no custom G4Allocator
    // (safe with G4String/std::string members under MT)

    G4int          fPlateIndex  = -1;
    G4int          fTrackID     = 0;
    G4int          fParentID    = 0;
    G4int          fPDG         = 0;
    std::string    fParticleName;
    std::string    fProcessName;
    G4double       fEdep        = 0.;
    G4double       fStepLength  = 0.;
    G4double       fKineticE    = 0.;
    G4double       fTime        = 0.;
    G4ThreeVector  fPosition;
    G4ThreeVector  fMomentumDir;
};

#endif
