// ==============================================================
// EventAction.hh — Per-event accumulation for ALL volumes
//
// Coincidence logic
// ─────────────────
// At the end of each event, the energy deposited in each of the
// four scintillator plates is compared against kDiscrimThreshold_MeV.
// A plate "fires" if its total edep exceeds the threshold.
// The firing pattern is stored as a 4-bit bitmask:
//
//   bit 0 = Scint_0  (+74 mm, top)
//   bit 1 = Scint_1  (+54 mm)
//   bit 2 = Scint_2  (−56 mm)
//   bit 3 = Scint_3  (−76 mm, bottom)
//
// All 15 non-zero patterns (2^4 − 1) are counted over the run.
// The four standard coincidence classes used in the real experiment
// are explicitly named:
//
//   2-fold  — any pair fires (used for plateau + coincidence sweep)
//   3-fold  — any three fire (used for flux measurement)
//   4-fold  — all four fire (through-going muon)
//   Efficiency of plate X = n(all fire) / n(all others fire)
//
// The per-event bitmask is written to coincidences.csv (no cap).
// Run-level counts are written to coincidence_summary.csv.
// ==============================================================

#ifndef EVENT_ACTION_HH
#define EVENT_ACTION_HH

#include "G4UserEventAction.hh"
#include "G4Types.hh"

#include <set>
#include <string>
#include <vector>

// Volume indices — one per physical component
enum VolIdx {
    kScint0 = 0, kScint1, kScint2, kScint3,
    kAlPlate, kCopper, kAirHole,
    kCeiling, kRoof, kWorld,
    kNVols
};

// Particle species bins
enum PartIdx {
    kMuMinus = 0, kMuPlus, kElectron, kPositron,
    kGamma, kOther, kAll,
    kNParts
};

// ── Discriminator threshold ───────────────────────────────────────────────────
// Typical plastic scintillator + PMT discriminator for cosmic muons.
// A MIP deposits ~2 MeV/cm; each plate is 2 cm → ~4 MeV.
// Threshold set conservatively at 0.5 MeV to accept stopping muons
// (which deposit less than MIP) and reject thermal noise.
// Units: MeV (matches fEdep[][]).
static constexpr double kDiscrimThreshold_MeV = 0.5;

struct ScintEntryRecord {
    G4int    vol;
    G4int    pdg;
    G4int    trackID;
    G4double ke_MeV;
};

struct DecayRecord {
    G4int    vol;
    G4int    pdg;
    G4int    trackID;
    G4double ke_MeV;
    G4double x_mm;
    G4double y_mm;
    G4double z_mm;
    G4double time_ns;
};

struct SecondaryEntryRecord {
    G4int    vol;
    G4int    pdg;
    G4int    trackID;
    G4double ke_MeV;
};

// Human-readable labels (defined in .cc)
extern const char* kVolNames[kNVols];
extern const char* kPartNames[kNParts];

VolIdx  VolumeNameToIdx(const std::string& name);
PartIdx PDGToPartIdx(G4int pdg);

class EventAction : public G4UserEventAction
{
public:
    EventAction();
    ~EventAction() override = default;

    void BeginOfEventAction(const G4Event*) override;
    void EndOfEventAction(const G4Event*)   override;

    void AccumulateStep(VolIdx vol, PartIdx part,
                        G4double edep, G4double stepLen);

    void RecordEntry(VolIdx vol, PartIdx part, G4int trackID);
    void RecordExit (VolIdx vol, PartIdx part, G4int trackID);
    void RecordScintEntry(VolIdx vol, G4int pdg, G4int trackID, G4double ke);

    void RecordDecay(VolIdx vol, G4int pdg, G4int trackID,
                     G4double ke, G4double x, G4double y,
                     G4double z, G4double time);

    bool IsAlDecaySecondary(G4int parentID) const {
        return fAlDecayMuonTrackIDs.count(parentID) > 0;
    }
    void RecordDecaySecEntry(VolIdx vol, PartIdx part, G4int trackID);
    void RecordDecaySecExit (VolIdx vol, PartIdx part, G4int trackID);
    void RecordDecaySecEntryKE(VolIdx vol, G4int pdg,
                                G4int trackID, G4double ke);

private:
    G4double fEdep   [kNVols][kNParts] = {};
    G4double fStepLen[kNVols][kNParts] = {};

    std::vector<ScintEntryRecord>     fScintEntries;
    std::vector<DecayRecord>          fDecayRecords;
    std::vector<SecondaryEntryRecord> fDecaySecEntryKEs;

    std::set<G4int> fEntered[kNVols][kNParts];
    std::set<G4int> fExited [kNVols][kNParts];

    std::set<G4int> fAlDecayMuonTrackIDs;

    std::set<G4int> fDecSecEntered[kNVols][kNParts];
    std::set<G4int> fDecSecExited [kNVols][kNParts];

    G4int fHCIDs[4] = {-1, -1, -1, -1};

    bool fHitDetector = false;
    std::vector<std::string> fTrajBuffer;

public:
    void BufferTrajLine(const std::string& line) { fTrajBuffer.push_back(line); }
    void SetHitDetector() { fHitDetector = true; }
};

#endif
