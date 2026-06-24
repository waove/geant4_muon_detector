// ==============================================================
// DetectorConstruction.hh
//
// Translates the detector geometry into C++ code and sets up
// the local uniform magnetic field inside the copper coil.
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

    // Magnetic field object — owned by the field manager after assignment,
    // but kept here so the destructor can clean up if needed.
    G4UniformMagField* fMagField  = nullptr;
};

#endif
