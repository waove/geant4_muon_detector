// ==============================================================
// DetectorConstruction.hh — Stripped-down geometry
//
// Copper coil + air hole + 4 scintillator plates + Al absorber.
// Concrete ceiling/roof removed — source injects directly above
// the coil from a pre-simulated muon file.
// Local uniform B field inside the coil for spin precession.
// ==============================================================

#ifndef DETECTOR_CONSTRUCTION_HH
#define DETECTOR_CONSTRUCTION_HH

#include "G4VUserDetectorConstruction.hh"
#include "G4LogicalVolume.hh"
#include "G4UniformMagField.hh"

class DetectorConstruction : public G4VUserDetectorConstruction
{
public:
    DetectorConstruction();
    ~DetectorConstruction() override = default;

    G4VPhysicalVolume* Construct() override;
    void ConstructSDandField() override;

    G4LogicalVolume* GetScintLV(int idx) const { return fScintLV[idx]; }
    G4LogicalVolume* GetAlPlateLV()      const { return fAlPlateLV; }

private:
    G4LogicalVolume* fScintLV[4]  = {};
    G4LogicalVolume* fAlPlateLV   = nullptr;
    G4LogicalVolume* fCopperLV    = nullptr;
    G4LogicalVolume* fAirHoleLV   = nullptr;

    G4UniformMagField* fMagField  = nullptr;
};

#endif
