// ==============================================================
// PrimaryGeneratorAction.cc — MAGNETIC FIELD TEST MODE
//
// Low-energy electron source in the air gap between the Al plate
// and Scint 2, inside the copper coil where the B-field lives.
//
// Geometry (x-axis):
//   Al plate:   [−1, +9] mm
//   Air gap:    [−46, −1] mm   ← SOURCE HERE  (45 mm wide)
//   Scint 2:    [−66, −46] mm
//
//   Air hole y-extent: ±314 mm  (628 mm travel distance)
//   Air hole z-extent: ±360 mm
//
// Physics:
//   B = 1 T along +z   (UNPHYSICAL — test only)
//   Larmor radius  r = p / (0.3 × B[T])  [metres, p in GeV/c]
//
//   E_kin = 0.5 MeV → r ≈  2.5 mm  (diameter   5 mm)
//   E_kin = 1.0 MeV → r ≈  4.7 mm  (diameter   9 mm)
//   E_kin = 2.0 MeV → r ≈  8.2 mm  (diameter  16 mm)
//   E_kin = 3.0 MeV → r ≈ 11.6 mm  (diameter  23 mm)
//
//   All diameters fit inside the 45 mm air gap.
//
// Direction: primarily along +y → electrons spiral in x-y plane
//   (circular motion ⊥ B) while drifting along y (628 mm of room).
//   Small z component → helix pitch along B visible in y-z view.
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

PrimaryGeneratorAction::PrimaryGeneratorAction()
    : fParticleGun(new G4ParticleGun(1))
{
    auto* table = G4ParticleTable::GetParticleTable();
    fParticleGun->SetParticleDefinition(table->FindParticle("e-"));

    G4cout << "\n  ============================================" << G4endl;
    G4cout << "  MAGNETIC FIELD TEST MODE" << G4endl;
    G4cout << "  Source: low-energy electrons (0.5–3 MeV)" << G4endl;
    G4cout << "  Position: air gap x ≈ −23 mm" << G4endl;
    G4cout << "            (between Al plate and Scint 2)" << G4endl;
    G4cout << "  B-field: 1 T along +z (LOCAL to air hole)" << G4endl;
    G4cout << "  Expected Larmor radii: 2.5–12 mm" << G4endl;
    G4cout << "  ============================================\n" << G4endl;
}

PrimaryGeneratorAction::~PrimaryGeneratorAction()
{
    delete fParticleGun;
}

void PrimaryGeneratorAction::GeneratePrimaries(G4Event* event)
{
    // ── Energy: uniform in [0.5, 3.0] MeV ──────────────────────
    G4double energy = (0.5 + 2.5 * G4UniformRand()) * MeV;
    fParticleGun->SetParticleEnergy(energy);

    // ── Position: centre of air gap between Al plate and Scint 2 ─
    // Al plate edge at x = −1 mm, Scint 2 edge at x = −46 mm.
    // Centre of gap at x = −23.5 mm.  Small random offset so
    // tracks don't all start from the same point.
    G4double x0 = -23.5 * mm + (G4UniformRand() - 0.5) * 10.0 * mm;  // ±5 mm
    G4double y0 = (G4UniformRand() - 0.5) * 100.0 * mm;  // ±50 mm near y-centre
    G4double z0 = (G4UniformRand() - 0.5) * 100.0 * mm;  // ±50 mm near z-centre
    fParticleGun->SetParticlePosition({x0, y0, z0});

    // ── Direction: primarily along +y ───────────────────────────
    // B is along +z → Lorentz force curves e⁻ in the x-y plane.
    // The electron drifts along y while spiralling.
    // Small x component seeds the circular motion;
    // small z component gives helical pitch along B.
    G4double dx = (G4UniformRand() - 0.5) * 0.4;   // small x-kick
    G4double dy = 1.0;                                // primary direction
    G4double dz = (G4UniformRand() - 0.5) * 0.3;   // small z for helix pitch

    G4double norm = std::sqrt(dx*dx + dy*dy + dz*dz);
    fParticleGun->SetParticleMomentumDirection({dx/norm, dy/norm, dz/norm});

    fParticleGun->GeneratePrimaryVertex(event);

    // ── Record gun energy ───────────────────────────────────────
    G4int eventID = event->GetEventID();
    auto* runAction = static_cast<const RunAction*>(
        G4RunManager::GetRunManager()->GetUserRunAction());

    if (runAction->ShouldRecordGunEnergy(eventID)) {
        runAction->GetGunEnergyStream()
            << eventID   << ","
            << 11        << ","
            << energy / MeV << ","
            << dz / norm << "\n";
    }
}
