// ==============================================================
// DetectorConstruction.cc — Geometry for lifetime + precession
//
// Changes from previous version:
//   1. Plates 0+1 and 2+3 are spaced 4 mm apart (were touching)
//      → decay electrons no longer always fire both plates of a pair
//   2. Plate 3 moved to x = −1000 mm (world volume, outside coil)
//      → narrow angular acceptance → large precession asymmetry A
//   3. Scintillator Z half-length trimmed to 350 mm (was 500 mm)
//      → fits inside the air hole (±360 mm) without overlap
//   4. World extended to ±1200 mm in x to accommodate distant plate
//
// Geometry layout (x is vertical, muons from +x):
//
//   x [mm]
//   +100   source plane
//   +95    copper top
//   +78    ┌ Plate 0 (20 mm) ┐  ← START trigger pair
//   +52    └ Plate 1 (20 mm) ┘     4 mm gap between them
//   +4     Al absorber (10 mm)  ← muons stop here
//   −56    Plate 2 (20 mm)     ← nearby VETO / local stop
//   −95    copper bottom
//   −1000  Plate 3 (20 mm)     ← distant stop (precession)
//
// B field: 40 mT along +z inside the air hole.
//
// ★ CRITICAL: SPIN PHYSICS REQUIRED ★
// Add to your physics list construction:
//
//   #include "G4SpinDecayPhysics.hh"
//   ...
//   RegisterPhysics(new G4SpinDecayPhysics());
//
// Without this, G4Decay produces isotropic muon decay — the spin
// polarization you set on the gun has NO effect on the decay electron
// direction.  G4SpinDecayPhysics replaces G4Decay with G4DecayWithSpin
// and registers G4MuonDecayChannelWithSpin, which correlates the
// electron emission direction with the instantaneous spin vector.
// The spin precesses in the B field via G4Spin transport.
// ==============================================================

#include "DetectorConstruction.hh"

#include "G4NistManager.hh"
#include "G4Box.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4SystemOfUnits.hh"
#include "G4VisAttributes.hh"
#include "G4Colour.hh"
#include "G4UserLimits.hh"

#include "G4UniformMagField.hh"
#include "G4FieldManager.hh"
#include "G4Mag_UsualEqRhs.hh"
#include "G4ClassicalRK4.hh"
#include "G4ChordFinder.hh"

// 40 mT = 400 Gauss, along +z (transverse to beam → spin precession in x-y)
static constexpr G4double kFieldStrength = 0.004 * tesla;
static constexpr G4double kMinStep       = 0.1  * mm;

DetectorConstruction::DetectorConstruction() {}

G4VPhysicalVolume* DetectorConstruction::Construct()
{
    auto* nist = G4NistManager::Instance();

    G4Material* air    = nist->FindOrBuildMaterial("G4_AIR");
    G4Material* copper = nist->FindOrBuildMaterial("G4_Cu");
    G4Material* alumin = nist->FindOrBuildMaterial("G4_Al");
    G4Material* scint  = nist->FindOrBuildMaterial("G4_PLASTIC_SC_VINYLTOLUENE");

    // ── World ─────────────────────────────────────────────────────────────
    // Extended in −x to accommodate the distant plate at −1000 mm.
    // Asymmetric: +200 mm (source side), −1200 mm (distant plate side).
    // Centre shifted so plate 3 fits.
    auto* worldS  = new G4Box("World", 1200*mm, 500*mm, 600*mm);
    auto* worldLV = new G4LogicalVolume(worldS, air, "World");
    worldLV->SetVisAttributes(G4VisAttributes::GetInvisible());
    auto* worldPV = new G4PVPlacement(nullptr, {}, worldLV, "World",
                                       nullptr, false, 0, true);

    // ── Copper coil (outer shell) ─────────────────────────────────────────
    auto* copperS = new G4Box("Shape1", 95*mm, 315*mm, 360*mm);
    fCopperLV = new G4LogicalVolume(copperS, copper, "Shape1");
    fCopperLV->SetVisAttributes(G4VisAttributes(G4Colour(0.9, 0.1, 0.1)));
    new G4PVPlacement(nullptr, {0,0,0}, fCopperLV, "Shape1",
                      worldLV, false, 0, true);

    // ── Air hole (coil interior) ──────────────────────────────────────────
    auto* airS = new G4Box("Shape2", 94*mm, 314*mm, 360*mm);
    fAirHoleLV = new G4LogicalVolume(airS, air, "Shape2");
    fAirHoleLV->SetVisAttributes(G4VisAttributes::GetInvisible());
    new G4PVPlacement(nullptr, {0,0,0}, fAirHoleLV, "Shape2",
                      fCopperLV, false, 0, true);

    // ── Scintillator plates ───────────────────────────────────────────────
    //
    // Z half-length: 350 mm (fits inside air hole ±360 mm with margin).
    // Previous value of 500 mm protruded into the copper walls — Geant4
    // doesn't clip daughters, so those regions were silently overlapping.
    //
    const G4double scintHalfX = 10*mm;
    const G4double scintHalfY = 314*mm;
    const G4double scintHalfZ = 350*mm;

    auto* scintS = new G4Box("Plate_scin", scintHalfX, scintHalfY, scintHalfZ);

    // Plates 0–2 are inside the air hole.
    // Plate 3 is the distant plate (placed in world volume below).
    //
    //   Plate 0: +78 mm  ← was +74 mm (shifted +4 to open gap with plate 1)
    //   Plate 1: +52 mm  ← was +54 mm (shifted −2)
    //        gap: 78−10 = 68 top edge of 1; 52+10 = 62 bottom edge of 0
    //             → 68 − 62 = 6 mm gap (electron can stop between them)
    //   Plate 2: −58 mm  ← standalone VETO / nearby stop
    //
    const G4double xPosInner[3] = { +78*mm, +52*mm, -58*mm };

    for (int i = 0; i < 3; ++i) {
        G4String name = "Plate_scin_" + std::to_string(i);
        fScintLV[i] = new G4LogicalVolume(scintS, scint, name);
        fScintLV[i]->SetVisAttributes(G4VisAttributes(G4Colour(0.1, 0.9, 0.1)));
        new G4PVPlacement(nullptr, {xPosInner[i], 0, 0},
                          fScintLV[i], name,
                          fAirHoleLV, false, i, true);
    }

    // ── Plate 3: distant stop detector for precession ─────────────────────
    //
    // Placed at x = −1000 mm in the WORLD volume (outside the coil).
    // From the Al absorber (x ≈ +4 mm) this is ~1004 mm away.
    // Solid angle ≈ (628 × 700) / 1004² ≈ 0.44 sr → about 3.5% of 4π.
    // This gives strong directional selectivity: the detector only sees
    // electrons emitted in a narrow cone around −x, so the precession
    // asymmetry A approaches the theoretical maximum of ~1/3.
    //
    // Adjust this distance to trade off A² × N:
    //   −500 mm: more counts, smaller A  (~8% of 4π)
    //   −1000 mm: fewer counts, larger A  (~3.5% of 4π)
    //   −2000 mm: very few counts, A near max  (~1% of 4π)
    //
    static constexpr G4double kDistantPlateX = -1000.0 * mm;

    G4String distName = "Plate_scin_3";
    fScintLV[3] = new G4LogicalVolume(scintS, scint, distName);
    fScintLV[3]->SetVisAttributes(G4VisAttributes(G4Colour(0.9, 0.9, 0.1)));
    new G4PVPlacement(nullptr, {kDistantPlateX, 0, 0},
                      fScintLV[3], distName,
                      worldLV, false, 3, true);

    G4cout << "\n  Distant plate 3 at x = " << kDistantPlateX / mm
           << " mm (in world volume, outside B field)" << G4endl;

    // ── Aluminium absorber plate ──────────────────────────────────────────
    auto* alS = new G4Box("Plate_Al", 5*mm, 314*mm, 350*mm);
    fAlPlateLV = new G4LogicalVolume(alS, alumin, "Plate_Al");
    fAlPlateLV->SetVisAttributes(G4VisAttributes(G4Colour(0.1, 0.1, 0.9)));
    new G4PVPlacement(nullptr, {4*mm, 0, 0},
                      fAlPlateLV, "Plate_Al",
                      fAirHoleLV, false, 0, true);

    // ── Step limits ───────────────────────────────────────────────────────
    // Fine steps inside the B field region for accurate spin transport.
    // The distant plate doesn't need fine steps (no field there).
    auto* fieldLimits = new G4UserLimits(0.5*mm);
    fCopperLV ->SetUserLimits(fieldLimits);
    fAirHoleLV->SetUserLimits(fieldLimits);
    for (int i = 0; i < 3; ++i) fScintLV[i]->SetUserLimits(fieldLimits);
    fAlPlateLV->SetUserLimits(fieldLimits);

    // Coarser steps for the distant plate (no field, just needs boundary hit)
    auto* distantLimits = new G4UserLimits(2.0*mm);
    fScintLV[3]->SetUserLimits(distantLimits);

    return worldPV;
}

// ==============================================================
// ConstructSDandField — local B field inside the air hole
//
// The field is LOCAL to the air hole only.  Plate 3 (distant) is
// in the world volume and sees zero field.  This is physically
// correct: the solenoid field is confined to the coil interior.
// ==============================================================
void DetectorConstruction::ConstructSDandField()
{
    fMagField = new G4UniformMagField(
        G4ThreeVector(0., 0., kFieldStrength));

    auto* equation    = new G4Mag_UsualEqRhs(fMagField);
    auto* stepper     = new G4ClassicalRK4(equation);
    auto* chordFinder = new G4ChordFinder(fMagField, kMinStep, stepper);

    auto* localFM = new G4FieldManager(fMagField, chordFinder);
    localFM->SetDeltaIntersection(0.01 * mm);
    localFM->SetDeltaOneStep     (0.01 * mm);

    fAirHoleLV->SetFieldManager(localFM, /*forceToAllDaughters=*/true);

    G4cout << "\n  === Magnetic field ==="
           << "\n  B = (0, 0, " << kFieldStrength/tesla << " T)"
           << "  applied to air hole + daughters (plates 0-2, Al)"
           << "\n  Plate 3 (distant): no field (world volume)"
           << G4endl;
}