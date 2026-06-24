// ==============================================================
// ScintillatorSD.hh — Sensitive detector for scintillator plates
//
// THIS IS WHERE YOU GET FULL PARTICLE-TRACKING CONTROL.
//
// ProcessHits() is called for every step inside the scintillator.
// You have access to:
//   - Particle type, PDG code, track ID, parent ID
//   - Energy deposit, step length, kinetic energy
//   - Pre/post step position and momentum
//   - Creator process name (e.g. "muIoni", "eBrem", "Decay")
//   - Volume copy number (which scintillator plate)
//
// The hit collection stores ScintHit objects that accumulate
// whatever per-step data you want. At EndOfEvent you can write
// them to file, fill histograms, apply coincidence logic, etc.
// ==============================================================

#ifndef SCINTILLATOR_SD_HH
#define SCINTILLATOR_SD_HH

#include "G4VSensitiveDetector.hh"
#include "G4THitsCollection.hh"
#include "ScintHit.hh"

class ScintillatorSD : public G4VSensitiveDetector
{
public:
    ScintillatorSD(const G4String& name, G4int plateIndex);
    ~ScintillatorSD() override = default;

    void   Initialize(G4HCofThisEvent* hce) override;
    G4bool ProcessHits(G4Step* step, G4TouchableHistory*) override;
    void   EndOfEvent(G4HCofThisEvent* hce) override;

    G4int GetPlateIndex() const { return fPlateIndex; }

private:
    using ScintHitsCollection = G4THitsCollection<ScintHit>;

    ScintHitsCollection* fHitsCollection = nullptr;
    G4int                fHCID           = -1;
    G4int                fPlateIndex;       // 0–3
};

#endif
