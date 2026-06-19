// ==============================================================
// PrimaryGeneratorAction.hh — CSV-sampled muon source
//
// Reads a pre-simulated muon file (CSV) containing muons that
// already passed through the concrete overburden.  Each event
// picks a random row and injects the muon just above the coil.
//
// CSV format (one muon per row):
//   eventID,trackID,pdg,ke_MeV,px,py,pz,polx,poly,polz
//
// Polarization is preserved — essential for spin precession /
// lifetime asymmetry measurements in the magnetic field.
// ==============================================================

#ifndef PRIMARY_GENERATOR_ACTION_HH
#define PRIMARY_GENERATOR_ACTION_HH

#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4ParticleGun.hh"
#include "globals.hh"

#include <vector>
#include <string>

class G4Event;

struct MuonRecord {
    G4int    pdg;
    G4double ke_MeV;
    G4double px, py, pz;       // momentum direction (unit vector)
    G4double polx, poly, polz; // polarization vector
};

class PrimaryGeneratorAction : public G4VUserPrimaryGeneratorAction
{
public:
    explicit PrimaryGeneratorAction(const std::string& csvPath = "muon_source.csv");
    ~PrimaryGeneratorAction() override;

    void GeneratePrimaries(G4Event* event) override;

private:
    void LoadCSV(const std::string& path);

    G4ParticleGun*          fGun;
    std::vector<MuonRecord> fRecords;
};

#endif
