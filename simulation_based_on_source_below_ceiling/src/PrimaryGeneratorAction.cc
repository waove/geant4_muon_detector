// ==============================================================
// PrimaryGeneratorAction.cc — CSV-sampled muon source
//
// Reads a pre-simulated CSV of muons at the detector level.
// Each event randomly samples one row (with replacement) and
// injects the muon just above the copper coil at x = +100 mm,
// with y,z uniformly distributed over the air hole face.
//
// CSV format (header row skipped):
//   eventID,trackID,pdg,ke_MeV,px,py,pz,polx,poly,polz
//
// Polarization is set on the gun — critical for spin precession
// measurements with the transverse B field.
// ==============================================================

#include "PrimaryGeneratorAction.hh"

#include "G4ParticleTable.hh"
#include "G4SystemOfUnits.hh"
#include "Randomize.hh"

#include <fstream>
#include <sstream>
#include <stdexcept>

PrimaryGeneratorAction::PrimaryGeneratorAction(const std::string& csvPath)
    : fGun(new G4ParticleGun(1))
{
    LoadCSV(csvPath);
    if (fRecords.empty())
        throw std::runtime_error(
            "PrimaryGeneratorAction: no muon records loaded from " + csvPath);

    G4cout << "PrimaryGeneratorAction: loaded " << fRecords.size()
           << " muon records from " << csvPath << G4endl;
}

PrimaryGeneratorAction::~PrimaryGeneratorAction()
{
    delete fGun;
}

void PrimaryGeneratorAction::LoadCSV(const std::string& path)
{
    std::ifstream file(path);
    if (!file.is_open())
        throw std::runtime_error(
            "PrimaryGeneratorAction: cannot open " + path);

    std::string line;
    // Skip header
    std::getline(file, line);
    int lineNum = 1; // Keep track of header (line 1)
    
    while (std::getline(file, line)) {
        lineNum++;
        
        // Skip purely empty lines or comments
        if (line.empty() || line[0] == '#') continue;

        std::istringstream ss(line);
        std::string token;
        MuonRecord rec{};

        try {
            // eventID (skip)
            if (!std::getline(ss, token, ',')) continue; 
            // trackID (skip)
            if (!std::getline(ss, token, ',')) continue;

            std::getline(ss, token, ','); rec.pdg    = std::stoi(token);
            std::getline(ss, token, ','); rec.ke_MeV = std::stod(token);
            std::getline(ss, token, ','); rec.px     = std::stod(token);
            std::getline(ss, token, ','); rec.py     = std::stod(token);
            std::getline(ss, token, ','); rec.pz     = std::stod(token);
            std::getline(ss, token, ','); rec.polx   = std::stod(token);
            std::getline(ss, token, ','); rec.poly   = std::stod(token);
            std::getline(ss, token, ','); rec.polz   = std::stod(token);

            fRecords.push_back(rec);
        } catch (const std::exception& e) {
            // This will print the exact line number and content causing the crash
            G4cerr << "\n[ERROR] PrimaryGeneratorAction: Failed to parse CSV at line " << lineNum 
                << "\n  Line content: '" << line << "'"
                << "\n  Exception: " << e.what() << "\n" << G4endl;
            throw std::runtime_error("CSV parsing failed.");
        }
    }
}

void PrimaryGeneratorAction::GeneratePrimaries(G4Event* event)
{
    // ── Pick a random record ──────────────────────────────────────────────
    G4int idx = static_cast<G4int>(
        G4UniformRand() * fRecords.size());
    if (idx >= (G4int)fRecords.size()) idx = fRecords.size() - 1;
    const MuonRecord& rec = fRecords[idx];

    // ── Particle type ─────────────────────────────────────────────────────
    auto* table = G4ParticleTable::GetParticleTable();
    auto* partDef = table->FindParticle(rec.pdg);
    if (!partDef) {
        G4cerr << "PrimaryGeneratorAction: unknown PDG " << rec.pdg
               << ", skipping event" << G4endl;
        return;
    }
    fGun->SetParticleDefinition(partDef);

    // ── Energy ────────────────────────────────────────────────────────────
    fGun->SetParticleEnergy(rec.ke_MeV * MeV);

    // ── Position: just above the coil, random y,z over air-hole face ─────
    // Air hole: ±314 mm in y, ±360 mm in z
    // Source plane: x = +100 mm  (coil top face at +95 mm)
    G4double y0 = (G4UniformRand() - 0.5) * 2.0 * 314.0 * mm;
    G4double z0 = (G4UniformRand() - 0.5) * 2.0 * 360.0 * mm;
    fGun->SetParticlePosition({100.0 * mm, y0, z0});

    // ── Momentum direction (from CSV, already a unit vector) ──────────────
    // fGun->SetParticleMomentumDirection({rec.px, rec.py, rec.pz});
    fGun->SetParticleMomentumDirection({-1, 0, 0});

    // ── Polarization — essential for spin precession / g-2 ────────────────
    // fGun->SetParticlePolarization({rec.polx, rec.poly, rec.polz});
    fGun->SetParticlePolarization({0.1, 0, 0});


    fGun->GeneratePrimaryVertex(event);
}
