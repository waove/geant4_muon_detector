#include "DetectorConstruction.hh"
#include "ScintillatorSD.hh"

#include "G4NistManager.hh"
#include "G4Box.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4SystemOfUnits.hh"
#include "G4VisAttributes.hh"
#include "G4Colour.hh"
#include "G4SDManager.hh"
#include "G4UserLimits.hh"

// ── Magnetic field headers ────────────────────────────────────────────────────
#include "G4UniformMagField.hh"
#include "G4FieldManager.hh"
#include "G4TransportationManager.hh"
#include "G4MagIntegratorStepper.hh"
#include "G4ClassicalRK4.hh"
#include "G4Mag_UsualEqRhs.hh"
#include "G4ChordFinder.hh"

// ── Field configuration ───────────────────────────────────────────────────────
//
// The copper block is a solenoid coil.  The muon beam enters along −x.
// The coil axis is along z (transverse to the beam) which produces spin
// precession in the x-y plane — the standard orientation for muon spin
// rotation / lifetime asymmetry measurements.
//
// To change direction: replace {0, 0, kFieldStrength} with e.g.
//   {kFieldStrength, 0, 0}  → along beam axis (axial, no precession)
//   {0, kFieldStrength, 0}  → transverse in y
//   {0, 0, kFieldStrength}  → transverse in z  (current setting)
//
// 40 mT = 400 Gauss — typical lab electromagnet.
// 40 Gauss (= 4 mT) would be too weak to meaningfully deflect GeV muons.
//
static constexpr G4double kFieldStrength = 0.040 * tesla;   // 40 mT = 400 Gauss

// Chord finder accuracy — controls how accurately curved tracks are followed.
// 0.1 mm is a good trade-off: tight enough for the ~200 mm coil, not so
// tight that it slows tracking dramatically.
static constexpr G4double kMinStep      = 0.1  * mm;

DetectorConstruction::DetectorConstruction() {}

G4VPhysicalVolume* DetectorConstruction::Construct()
{
    auto* nist = G4NistManager::Instance();

    // ---- Materials --------------------------------------------------
    G4Material* air      = nist->FindOrBuildMaterial("G4_AIR");
    G4Material* copper   = nist->FindOrBuildMaterial("G4_Cu");
    G4Material* alumin   = nist->FindOrBuildMaterial("G4_Al");
    G4Material* scint    = nist->FindOrBuildMaterial("G4_PLASTIC_SC_VINYLTOLUENE");
    G4Material* concrete = nist->FindOrBuildMaterial("G4_CONCRETE");

    // ---- World ------------------------------------------------------
    auto* worldS  = new G4Box("World", 8000*mm, 1500*mm, 1500*mm);
    auto* worldLV = new G4LogicalVolume(worldS, air, "World");
    worldLV->SetVisAttributes(G4VisAttributes::GetInvisible());
    auto* worldPV = new G4PVPlacement(nullptr, {}, worldLV, "World",
                                       nullptr, false, 0, true);

    // ---- Copper block / coil (Shape1) --------------------------------
    auto* shape1S  = new G4Box("Shape1", 95*mm, 315*mm, 360*mm);
    fCopperLV = new G4LogicalVolume(shape1S, copper, "Shape1");
    fCopperLV->SetVisAttributes(G4VisAttributes(G4Colour(0.9, 0.1, 0.1)));
    new G4PVPlacement(nullptr, {0,0,0}, fCopperLV, "Shape1",
                      worldLV, false, 0, true);

    // ---- Air hole (Shape2) inside copper ----------------------------
    // This is the interior of the coil where the field is uniform.
    auto* shape2S  = new G4Box("Shape2", 94*mm, 314*mm, 360*mm);
    fAirHoleLV = new G4LogicalVolume(shape2S, air, "Shape2");
    fAirHoleLV->SetVisAttributes(G4VisAttributes::GetInvisible());
    new G4PVPlacement(nullptr, {0,0,0}, fAirHoleLV, "Shape2",
                      fCopperLV, false, 0, true);

    // ---- Scintillator plates (inside air hole) ----------------------
    const G4double scintHalfX = 10*mm;
    const G4double scintHalfY = 314*mm;
    const G4double scintHalfZ = 500*mm;
    const G4double xPos[4] = { +74*mm, +54*mm, -56*mm, -76*mm };

    auto* scintS = new G4Box("Plate_scin", scintHalfX, scintHalfY, scintHalfZ);

    for (int i = 0; i < 4; ++i) {
        G4String name = "Plate_scin_" + std::to_string(i);
        fScintLV[i] = new G4LogicalVolume(scintS, scint, name);
        fScintLV[i]->SetVisAttributes(G4VisAttributes(G4Colour(0.1, 0.9, 0.1)));
        new G4PVPlacement(nullptr, {xPos[i], 0, 0},
                          fScintLV[i], name,
                          fAirHoleLV, false, i, true);
    }

    // ---- Aluminium plate --------------------------------------------
    auto* alS = new G4Box("Plate_Al", 5*mm, 314*mm, 500*mm);
    fAlPlateLV = new G4LogicalVolume(alS, alumin, "Plate_Al");
    fAlPlateLV->SetVisAttributes(G4VisAttributes(G4Colour(0.1, 0.1, 0.9)));
    new G4PVPlacement(nullptr, {4*mm, 0, 0},
                      fAlPlateLV, "Plate_Al",
                      fAirHoleLV, false, 0, true);

    // ---- Step limits inside the detector enclosure ------------------
    // The chord finder already enforces small steps in the field region,
    // but we keep the explicit limit for the detailed trajectory events.
    auto* detLimits = new G4UserLimits(0.5*mm);
    fCopperLV ->SetUserLimits(detLimits);
    fAirHoleLV->SetUserLimits(detLimits);
    for (int i = 0; i < 4; ++i) fScintLV[i]->SetUserLimits(detLimits);
    fAlPlateLV->SetUserLimits(detLimits);

    // ---- Concrete ceiling & roof ------------------------------------
    auto* ceilS = new G4Box("Ceiling", 100*mm, 1500*mm, 1500*mm);
    auto* ceilLV = new G4LogicalVolume(ceilS, concrete, "Ceiling");
    ceilLV->SetVisAttributes(G4VisAttributes(G4Colour(0.5, 0.5, 0.5)));
    new G4PVPlacement(nullptr, {2685*mm, 0, 0},
                      ceilLV, "Ceiling", worldLV, false, 0, true);

    auto* roofS = new G4Box("Roof", 200*mm, 1500*mm, 1500*mm);
    auto* roofLV = new G4LogicalVolume(roofS, concrete, "Roof");
    roofLV->SetVisAttributes(G4VisAttributes(G4Colour(0.5, 0.5, 0.5)));
    new G4PVPlacement(nullptr, {5985*mm, 0, 0},
                      roofLV, "Roof", worldLV, false, 0, true);

    return worldPV;
}

// ==============================================================
// ConstructSDandField
//
// Called once per worker thread after Construct().
// Attaches sensitive detectors and sets up the local uniform
// magnetic field inside the copper coil (air hole volume).
//
// Field is LOCAL to the air hole — it does not propagate into
// the world volume or the concrete ceiling/roof.  Daughter
// volumes (scintillators, Al plate) inherit it automatically
// because they are placed inside fAirHoleLV.
// ==============================================================
void DetectorConstruction::ConstructSDandField()
{
    // ── Sensitive detectors ───────────────────────────────────────────────────
    auto* sdManager = G4SDManager::GetSDMpointer();

    for (int i = 0; i < 4; ++i) {
        G4String sdName = "ScintSD_" + std::to_string(i);
        auto* sd = new ScintillatorSD(sdName, i);
        sdManager->AddNewDetector(sd);
        fScintLV[i]->SetSensitiveDetector(sd);
    }

    // ── Magnetic field ────────────────────────────────────────────────────────
    //
    // G4ThreeVector convention: (Bx, By, Bz) in the global frame.
    // Current setting: B along +z (transverse to −x beam direction).
    //
    fMagField = new G4UniformMagField(
        G4ThreeVector(0., 0., kFieldStrength));

    G4cout << "\n  === Magnetic field ===" << G4endl
           << "  Direction: +z  (transverse to beam)" << G4endl
           << "  Strength:  " << kFieldStrength / tesla << " mT  ("
           << kFieldStrength / gauss << " Gauss)" << G4endl;

    // ── Equation of motion + stepper + chord finder ───────────────────────────
    // G4Mag_UsualEqRhs  — standard Lorentz force equation for charged particles
    // G4ClassicalRK4    — 4th-order Runge-Kutta, good accuracy/speed balance
    // G4ChordFinder     — finds intersection of curved track with boundaries
    //
    auto* equation   = new G4Mag_UsualEqRhs(fMagField);
    auto* stepper    = new G4ClassicalRK4(equation);
    auto* chordFinder = new G4ChordFinder(fMagField, kMinStep, stepper);

    // ── Local field manager attached to the air hole ──────────────────────────
    // forceToAllDaughters = true → scintillators and Al plate inherit the field
    // without needing their own field managers.
    //
    auto* localFM = new G4FieldManager(fMagField, chordFinder);

    // Accuracy parameters for curved track following:
    // Delta intersection: max error on the intersection point [mm]
    // Delta one step:     max error per step [mm]
    localFM->SetDeltaIntersection(0.01 * mm);
    localFM->SetDeltaOneStep    (0.01 * mm);

    fAirHoleLV->SetFieldManager(localFM, /*forceToAllDaughters=*/true);

    G4cout << "  Applied to: air hole (interior of copper coil)\n"
           << "  Daughters (scintillators, Al plate) inherit field.\n"
           << "  Chord finder min step: " << kMinStep / mm << " mm"
           << G4endl;
}
