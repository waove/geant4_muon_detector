#include <G4VUserDetectorConstruction.hh>
#include <G4GlobalMagFieldMessenger.hh>
#include <G4tgbVolumeMgr.hh>
#include <G4AutoDelete.hh>
#include <G4TScoreNtupleWriter.hh>
#include <G4RunManagerFactory.hh>
#include <G4PhysListFactory.hh>
#include <G4AnalysisManager.hh>
#include <G4ScoringManager.hh>
#include <G4VisExecutive.hh>
#include <G4UIExecutive.hh>
#include <G4UImanager.hh>
#include <G4VUserPrimaryGeneratorAction.hh>
#include <G4GeneralParticleSource.hh>
#include <G4UserSteppingAction.hh>
#include <G4Step.hh>
#include <G4UserRunAction.hh>
#include <G4VUserActionInitialization.hh>
#include <G4UserTrackingAction.hh>
#include <G4Track.hh>
#include <set>
#include <map>
#include <string>
#include <cstdlib>
#include "G4OpticalPhysics.hh"
#include "G4MaterialPropertiesTable.hh"
#include "G4MaterialTable.hh"
#include "G4Material.hh"
#include "G4NistManager.hh"
#include "G4LogicalVolumeStore.hh"
#include "G4OpticalSurface.hh"
#include "G4LogicalBorderSurface.hh"
#include "G4LogicalSkinSurface.hh"
#include "G4SystemOfUnits.hh"
#include "G4OpticalPhoton.hh"
#include "G4Trap.hh"
#include "G4Box.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4VisAttributes.hh"
#include "G4Color.hh"
#include "G4UserEventAction.hh"
#include "G4Event.hh"
#include <random>

// ================================================================
// GLOBAL TOGGLE — read once at startup in main()
// Set ENABLE_PATH_TRACKING=1 to enable photon path recording.
// Default is OFF (0) for fast production runs.
//
// Usage:
//   ENABLE_PATH_TRACKING=1 ./mingle_combined run.mac   # paths ON
//   ENABLE_PATH_TRACKING=0 ./mingle_combined run.mac   # paths OFF (default)
// ================================================================
static bool gPathTrackingEnabled  = false;
static bool gParentageEnabled     = false;  // ENABLE_PARENTAGE=1 to record photon parentage

// ================================================================
// THREAD-LOCAL STATE
// ================================================================

// --- Particle name map (only needed for parentage) ---
static G4ThreadLocal std::map<G4int, G4String>* gTrackParticleMap    = nullptr;
static G4ThreadLocal std::map<G4int, G4String>* gPhotonParentNameMap = nullptr;

// --- Path tracking buffers (only used when gPathTrackingEnabled) ---
struct PhotonStep { G4double x, y, z, t; };

class PhotonTrackBuffer
{
public:
    void AddStep(G4double x, G4double y, G4double z, G4double t) {
        fSteps.push_back({x, y, z, t});
    }

    void FlushToNtuple(G4int ntuple, G4int trackID, G4int eventID) {
        auto am = G4AnalysisManager::Instance();
        for (auto& s : fSteps) {
            am->FillNtupleDColumn(ntuple, 0, s.x);
            am->FillNtupleDColumn(ntuple, 1, s.y);
            am->FillNtupleDColumn(ntuple, 2, s.z);
            am->FillNtupleDColumn(ntuple, 3, s.t);
            am->FillNtupleIColumn(ntuple, 4, trackID);
            am->FillNtupleIColumn(ntuple, 5, eventID);
            am->AddNtupleRow(ntuple);
        }
    }

    void Clear() { fSteps.clear(); }

private:
    std::vector<PhotonStep> fSteps;
};

// Ntuple indices — assigned dynamically based on whether path tracking is on
// 0: MuonEntry
// 1: PhotonHits
// 2: PhotonsProduced
// 3: PhotonParentage
// 4: PhotonPaths        (only if path tracking enabled)
// 5: LostPhotonPaths    (only if path tracking enabled)
static const G4int NTUPLE_MUON       = 0;
static const G4int NTUPLE_HITS       = 1;
static const G4int NTUPLE_PRODUCED   = 2;
static const G4int NTUPLE_PARENTAGE  = 3;
static const G4int NTUPLE_PATHS      = 4;
static const G4int NTUPLE_LOST       = 5;

static G4ThreadLocal std::map<G4int, PhotonTrackBuffer>* gPhotonBuffers  = nullptr;
static G4ThreadLocal std::map<G4int, PhotonTrackBuffer>* gLostBuffers    = nullptr;
static G4ThreadLocal G4int*                              gLostCount       = nullptr;
static G4ThreadLocal std::set<G4int>*                    gHandledTracks   = nullptr;
static const G4int MAX_LOST = 50;

// ================================================================
// DETECTOR
// ================================================================
class Detector : public G4VUserDetectorConstruction
{
public:
  G4VPhysicalVolume* Construct() override {

    G4NistManager* nist = G4NistManager::Instance();

    // ── Geometry parameters ───────────────────────────────────
    G4double scintHalfX =  10.0*mm;
    G4double scintHalfY = 314.0*mm;
    G4double scintHalfZ = 500.0*mm;
    G4double lgHalfLen  = 200.0*mm;
    G4double pmtHalfW   =  30.0*mm;
    G4double pmtHalfX   =  10.0*mm;
    G4double pmtHalfY   =  30.0*mm;
    G4double pmtHalfZ   =   5.0*mm;
    G4double worldHalfX =  80.0*mm;
    G4double worldHalfY = 400.0*mm;
    G4double worldHalfZ = 1000.0*mm;

    // ── Materials ─────────────────────────────────────────────
    G4Material* vacuum   = nist->FindOrBuildMaterial("G4_Galactic");
    G4Material* scintMat = nist->FindOrBuildMaterial("G4_PLASTIC_SC_VINYLTOLUENE");
    G4Material* acrylic  = nist->FindOrBuildMaterial("G4_PLEXIGLASS");
    G4Material* pmtMat   = nist->FindOrBuildMaterial("G4_Galactic");

    // ── Optical properties ────────────────────────────────────
    std::vector<G4double> energies = {
        2.48*eV, 2.58*eV, 2.70*eV, 2.82*eV,
        2.92*eV, 3.02*eV, 3.18*eV, 3.35*eV, 3.54*eV
    };
    std::vector<G4double> scintRIndex(9, 1.58);
    std::vector<G4double> scintAbsLength = {
        380*cm, 380*cm, 380*cm, 375*cm,
        370*cm, 350*cm, 300*cm, 180*cm, 50*cm
    };

    std::vector<G4double> scintRayleigh = {
    2000*cm, 1800*cm, 1500*cm, 1200*cm,
    1000*cm,  800*cm,  500*cm,  200*cm, 50*cm
    };

    std::vector<G4double> scintSpectrum = {
        0.08, 0.18, 0.40, 0.80, 1.00, 0.50, 0.10, 0.04, 0.01
    };

    G4MaterialPropertiesTable* scintMPT = new G4MaterialPropertiesTable();
    scintMPT->AddProperty("RINDEX",                  energies, scintRIndex);
    scintMPT->AddProperty("ABSLENGTH",               energies, scintAbsLength);
    scintMPT->AddProperty("RAYLEIGH",                energies, scintRayleigh);
    scintMPT->AddProperty("SCINTILLATIONCOMPONENT1", energies, scintSpectrum);
    scintMPT->AddConstProperty("SCINTILLATIONYIELD",          10000./MeV);
    scintMPT->AddConstProperty("RESOLUTIONSCALE",             1.0);
    scintMPT->AddConstProperty("SCINTILLATIONTIMECONSTANT1",  2.1*ns);
    scintMPT->AddConstProperty("SCINTILLATIONYIELD1",         1.0);
    scintMat->SetMaterialPropertiesTable(scintMPT);

    std::vector<G4double> acrylicRIndex(9, 1.49);
    std::vector<G4double> acrylicAbsLength = {
        400*cm, 400*cm, 395*cm, 390*cm,
        385*cm, 370*cm, 300*cm, 150*cm, 30*cm
    };

    std::vector<G4double> acrylicRayleigh = {
    5000*cm, 4000*cm, 3000*cm, 2000*cm,
    1500*cm, 1000*cm,  500*cm,  200*cm, 50*cm
    };

    G4MaterialPropertiesTable* acrylicMPT = new G4MaterialPropertiesTable();
    acrylicMPT->AddProperty("RINDEX",    energies, acrylicRIndex);
    acrylicMPT->AddProperty("ABSLENGTH", energies, acrylicAbsLength);
    acrylicMPT->AddProperty("RAYLEIGH",  energies, acrylicRayleigh);
    acrylic->SetMaterialPropertiesTable(acrylicMPT);

    // ── Solids ────────────────────────────────────────────────
    G4Box*  solidWorld     = new G4Box("World", worldHalfX, worldHalfY, worldHalfZ);
    G4Box*  solidScint     = new G4Box("Scintillator", scintHalfX, scintHalfY, scintHalfZ);
    G4Trap* solidLG        = new G4Trap("LightGuide",
        lgHalfLen, 0.*deg, 0.*deg,
        scintHalfY, scintHalfX, scintHalfX, 0.*deg,
        pmtHalfW,   scintHalfX, scintHalfX, 0.*deg);

    G4double cathodeHalfX  = pmtHalfX;
    G4double cathodeHalfY  = 10.0*mm;
    G4double cathodeHalfZ  = pmtHalfZ;
    G4Box* solidCathode    = new G4Box("PMT_cathode",   cathodeHalfX, cathodeHalfY, cathodeHalfZ);
    G4Box* solidBackplate  = new G4Box("PMT_backplate", pmtHalfX,     pmtHalfY,     pmtHalfZ);

    // ── Logical volumes ───────────────────────────────────────
    G4LogicalVolume* logicWorld     = new G4LogicalVolume(solidWorld,    vacuum,   "World");
    G4LogicalVolume* logicScint     = new G4LogicalVolume(solidScint,    scintMat, "Scintillator");
    G4LogicalVolume* logicLG        = new G4LogicalVolume(solidLG,       acrylic,  "LightGuide");
    G4LogicalVolume* logicCathode   = new G4LogicalVolume(solidCathode,  pmtMat,   "PMT_cathode");
    G4LogicalVolume* logicBackplate = new G4LogicalVolume(solidBackplate,pmtMat,   "PMT_backplate");

    // ── Physical placement ────────────────────────────────────
    G4VPhysicalVolume* physWorld = new G4PVPlacement(
        nullptr, G4ThreeVector(0,0,0),
        logicWorld, "World", nullptr, false, 0, true);

    G4VPhysicalVolume* physScint = new G4PVPlacement(
        nullptr, G4ThreeVector(0,0,0),
        logicScint, "Scintillator", logicWorld, false, 0, true);

    G4VPhysicalVolume* physLG_pZ = new G4PVPlacement(
        nullptr, G4ThreeVector(0, 0, scintHalfZ + lgHalfLen),
        logicLG, "LightGuide_pZ", logicWorld, false, 0, true);

    G4double pmtZ = scintHalfZ + 2.*lgHalfLen + pmtHalfZ;
    new G4PVPlacement(
        nullptr, G4ThreeVector(0, 0, pmtZ),
        logicBackplate, "PMT_backplate", logicWorld, false, 0, true);
    new G4PVPlacement(
        nullptr, G4ThreeVector(0, 0, pmtZ - 1.0*mm),
        logicCathode, "PMT_cathode", logicWorld, false, 0, true);

    // ── Optical surfaces ──────────────────────────────────────

    // ================================================================
    // WRAPPING SURFACE — controlled by environment variables
    //
    // Environment variables:
    //   WRAPPING_FINISH        — surface finish (default: groundfrontpainted)
    //                            options: groundfrontpainted, polishedfrontpainted,
    //                                     groundbackpainted, polishedbackpainted,
    //                                     ground, polished
    //   WRAPPING_REFLECTIVITY  — reflectivity 0-1 (default: 0.98)
    //   WRAPPING_SIGMA         — SigmaAlpha roughness in radians (default: 0.1)
    //                            only applied for ground-type finishes
    //                            0.0 = perfectly smooth, 0.3 = very rough
    // ================================================================

    // --- Finish type ---
    const char* envFinish = std::getenv("WRAPPING_FINISH");
    G4OpticalSurfaceFinish finishVal = groundfrontpainted;
    G4bool isBackpainted = false;
    G4bool isGround      = true;
    if (envFinish) {
        std::string fs(envFinish);
        if      (fs == "groundfrontpainted")   { finishVal = groundfrontpainted;  isBackpainted = false; isGround = true;  }
        else if (fs == "polishedfrontpainted") { finishVal = polishedfrontpainted; isBackpainted = false; isGround = false; }
        else if (fs == "ground")               { finishVal = ground;               isBackpainted = false; isGround = true;  }
        else if (fs == "polished")             { finishVal = polished;             isBackpainted = false; isGround = false; }
        else if (fs == "groundbackpainted")    { finishVal = groundbackpainted;    isBackpainted = true;  isGround = true;  }
        else if (fs == "polishedbackpainted")  { finishVal = polishedbackpainted;  isBackpainted = true;  isGround = false; }
        else G4cerr << ">>> WARNING: Unknown finish '" << fs
                    << "', using groundfrontpainted <<<" << G4endl;
        G4cout << ">>> Wrapping finish:       " << fs << " <<<" << G4endl;
    }

    // --- Reflectivity ---
    G4double reflVal = 0.98;
    const char* envRefl = std::getenv("WRAPPING_REFLECTIVITY");
    if (envRefl) {
        reflVal = std::stod(envRefl);
        G4cout << ">>> Wrapping reflectivity: " << reflVal << " <<<" << G4endl;
    }

    // --- SigmaAlpha (only for ground finishes) ---
    G4double sigmaVal = 0.1;
    const char* envSigma = std::getenv("WRAPPING_SIGMA");
    if (envSigma) sigmaVal = std::stod(envSigma);

    // --- Build surface ---
    G4OpticalSurface* wrappingSurf = new G4OpticalSurface("WrappingSurface");
    // Set type based on finish — dielectric_metal for uncoated reflective surfaces
    // (ground, polished) where SigmaAlpha controls roughness, dielectric_dielectric
    // for painted surfaces where the paint layer handles the reflection model
    if (finishVal == ground || finishVal == polished) {
        wrappingSurf->SetType(dielectric_metal);
    } else {
        // groundfrontpainted, polishedfrontpainted,
        // groundbackpainted, polishedbackpainted
        wrappingSurf->SetType(dielectric_dielectric);
    }
    wrappingSurf->SetFinish(finishVal);
    wrappingSurf->SetModel(unified);

    if (isGround) {
        wrappingSurf->SetSigmaAlpha(sigmaVal);
        G4cout << ">>> Wrapping sigma alpha:  " << sigmaVal << " rad <<<" << G4endl;
    } else {
        G4cout << ">>> Wrapping sigma alpha:  not applied (polished finish) <<<" << G4endl;
    }

    // --- Properties table ---
    // Reflection is purely Lambertian (diffuse) — correct for Teflon/white paint
    // lobe=spike=backscatter=0 means 100% of reflected light is Lambertian
    G4double lobeFrac = 0.5;
    const char* envLobe = std::getenv("WRAPPING_LOBE");
    if (envLobe) {
        lobeFrac = std::stod(envLobe);
        G4cout << ">>> Wrapping specular lobe: " << lobeFrac << " <<<" << G4endl;
    }
    std::vector<G4double> specLobe(9, lobeFrac);
    std::vector<G4double> wrappingRefl(9, reflVal);
    std::vector<G4double> wrappingEff (9, 0.0);
    std::vector<G4double> zeroVec     (9, 0.0);   // spike, backscatter all zero

    G4MaterialPropertiesTable* wrappingMPT = new G4MaterialPropertiesTable();
    wrappingMPT->AddProperty("REFLECTIVITY",          energies, wrappingRefl);
    wrappingMPT->AddProperty("EFFICIENCY",            energies, wrappingEff);
    wrappingMPT->AddProperty("SPECULARLOBECONSTANT",  energies, specLobe);
    wrappingMPT->AddProperty("SPECULARSPIKECONSTANT", energies, zeroVec);
    wrappingMPT->AddProperty("BACKSCATTERCONSTANT",   energies, zeroVec);

    // For backpainted: add air gap RINDEX to surface table
    // Geant4 uses this to simulate TIR at the scintillator-air boundary
    // internally without needing a physical air volume
    if (isBackpainted) {
        std::vector<G4double> airRIndex(9, 1.0);
        wrappingMPT->AddProperty("RINDEX", energies, airRIndex);
        G4cout << ">>> Backpainted: air gap RINDEX=1.0 applied <<<" << G4endl;
    }

    wrappingSurf->SetMaterialPropertiesTable(wrappingMPT);
    new G4LogicalSkinSurface("WrappingSkin",   logicScint, wrappingSurf);
    new G4LogicalSkinSurface("LGWrappingSkin", logicLG,    wrappingSurf);

    // Scint-LG interface
    G4OpticalSurface* scintLGSurf = new G4OpticalSurface("ScintLGSurface");
    scintLGSurf->SetType(dielectric_dielectric);
    scintLGSurf->SetFinish(polished);
    scintLGSurf->SetModel(unified);
    new G4LogicalBorderSurface("ScintToLG_pZ", physScint,  physLG_pZ, scintLGSurf);
    new G4LogicalBorderSurface("LGToScint_pZ", physLG_pZ, physScint,  scintLGSurf);

    // Cathode
    G4OpticalSurface* cathodeSurf = new G4OpticalSurface("CathodeSurface");
    cathodeSurf->SetType(dielectric_metal);
    cathodeSurf->SetFinish(polished);
    cathodeSurf->SetModel(glisur);
    std::vector<G4double> cathodeEff(9, 1.0), cathodeRefl(9, 0.0);
    G4MaterialPropertiesTable* cathodeMPT = new G4MaterialPropertiesTable();
    cathodeMPT->AddProperty("EFFICIENCY",   energies, cathodeEff);
    cathodeMPT->AddProperty("REFLECTIVITY", energies, cathodeRefl);
    cathodeSurf->SetMaterialPropertiesTable(cathodeMPT);
    new G4LogicalSkinSurface("CathodeSkin", logicCathode, cathodeSurf);

    // Backplate
    G4OpticalSurface* backplateSurf = new G4OpticalSurface("BackplateSurface");
    backplateSurf->SetType(dielectric_metal);
    backplateSurf->SetFinish(polished);
    backplateSurf->SetModel(glisur);
    std::vector<G4double> bpEff(9, 0.0), bpRefl(9, 0.0);
    G4MaterialPropertiesTable* bpMPT = new G4MaterialPropertiesTable();
    bpMPT->AddProperty("EFFICIENCY",   energies, bpEff);
    bpMPT->AddProperty("REFLECTIVITY", energies, bpRefl);
    backplateSurf->SetMaterialPropertiesTable(bpMPT);
    new G4LogicalSkinSurface("BackplateSkin", logicBackplate, backplateSurf);

    // ── Visualisation ─────────────────────────────────────────
    logicWorld->SetVisAttributes(G4VisAttributes::GetInvisible());
    G4VisAttributes* sv = new G4VisAttributes(G4Color(0.1,0.9,0.1,0.3)); sv->SetVisibility(true); logicScint->SetVisAttributes(sv);
    G4VisAttributes* lv = new G4VisAttributes(G4Color(0.5,0.5,1.0,0.4)); lv->SetVisibility(true); logicLG->SetVisAttributes(lv);
    G4VisAttributes* bv = new G4VisAttributes(G4Color(0.4,0.4,0.4,0.9)); bv->SetVisibility(true); logicBackplate->SetVisAttributes(bv);
    G4VisAttributes* cv = new G4VisAttributes(G4Color(1.0,0.8,0.0,0.9)); cv->SetVisibility(true); logicCathode->SetVisAttributes(cv);

    return physWorld;
  }

  void ConstructSDandField() override {
    G4AutoDelete::Register(new G4GlobalMagFieldMessenger());
  }
};

// ================================================================
// GENERATOR
// ================================================================
class Generator : public G4VUserPrimaryGeneratorAction
{
  private: G4GeneralParticleSource* fGPS;
  public:
    Generator() : G4VUserPrimaryGeneratorAction() { fGPS = new G4GeneralParticleSource; }
    ~Generator() { delete fGPS; }
    void GeneratePrimaries(G4Event* evt) { fGPS->GeneratePrimaryVertex(evt); }
};

// ================================================================
// STEPPING ACTION
// ================================================================
class SteppingAction : public G4UserSteppingAction
{
private:
    struct ParentageRow {
        G4int    eventID;
        G4int    parentID;
        G4String parentName;
        G4String creatorProcess;
    };

    G4bool fPrimaryEntryRecorded = false;
    G4int  fPhotonsProduced      = 0;
    // Parentage members — only used when gParentageEnabled
    G4int  fPhotonsSeen          = 0;
    static const G4int RESERVOIR_SIZE = 100;
    std::vector<ParentageRow> fReservoir;
    std::mt19937 fRng{std::random_device{}()};

public:
    void Reset() {
        fPrimaryEntryRecorded = false;
        fPhotonsProduced      = 0;
        if (gParentageEnabled) {
            fPhotonsSeen = 0;
            fReservoir.clear();
        }
    }
    G4int GetPhotonsProduced() const { return fPhotonsProduced; }

    void FlushParentage() {
        if (!gParentageEnabled) return;
        auto am = G4AnalysisManager::Instance();
        for (auto& row : fReservoir) {
            am->FillNtupleIColumn(NTUPLE_PARENTAGE, 0, row.eventID);
            am->FillNtupleIColumn(NTUPLE_PARENTAGE, 1, row.parentID);
            am->FillNtupleSColumn(NTUPLE_PARENTAGE, 2, row.parentName);
            am->FillNtupleSColumn(NTUPLE_PARENTAGE, 3, row.creatorProcess);
            am->AddNtupleRow(NTUPLE_PARENTAGE);
        }
    }

    void UserSteppingAction(const G4Step* step) override {
        G4Track* track = step->GetTrack();
        auto am = G4AnalysisManager::Instance();

        // ── Muon entry (Ntuple 0) ──────────────────────────────
        if (track->GetTrackID() == 1 && !fPrimaryEntryRecorded) {
            G4VPhysicalVolume* preVol  = step->GetPreStepPoint()->GetPhysicalVolume();
            G4VPhysicalVolume* postVol = step->GetPostStepPoint()->GetPhysicalVolume();
            if (preVol == nullptr || postVol == nullptr) return;
            if (preVol->GetName() != "Scintillator" &&
                postVol->GetName() == "Scintillator") {
                G4double yEntry = step->GetPostStepPoint()->GetPosition().y();
                G4double zEntry = step->GetPostStepPoint()->GetPosition().z();
                G4int    evID   = G4RunManager::GetRunManager()->GetCurrentEvent()->GetEventID();
                am->FillNtupleDColumn(NTUPLE_MUON, 0, yEntry);
                am->FillNtupleDColumn(NTUPLE_MUON, 1, zEntry);
                am->FillNtupleIColumn(NTUPLE_MUON, 2, evID);
                am->AddNtupleRow(NTUPLE_MUON);
                fPrimaryEntryRecorded = true;
            }
        }

        if (track->GetDefinition() != G4OpticalPhoton::OpticalPhoton()) return;

        G4int trackID = track->GetTrackID();
        G4int eventID = G4RunManager::GetRunManager()->GetCurrentEvent()->GetEventID();

        // ── Photon creation: count + parentage ────────────────
        if (track->GetCurrentStepNumber() == 1) {
            G4VPhysicalVolume* vol = step->GetPreStepPoint()->GetPhysicalVolume();
            if (vol && vol->GetName() == "Scintillator") {
                fPhotonsProduced++;

                // Parentage reservoir sampling — skipped in production mode
                if (gParentageEnabled) {
                    G4int parentID = track->GetParentID();

                    G4String creatorProcess = "unknown";
                    if (track->GetCreatorProcess())
                        creatorProcess = track->GetCreatorProcess()->GetProcessName();

                    G4String parentName = "unknown";
                    if (gPhotonParentNameMap) {
                        auto it = gPhotonParentNameMap->find(trackID);
                        if (it != gPhotonParentNameMap->end())
                            parentName = it->second;
                    }

                    fPhotonsSeen++;
                    if ((G4int)fReservoir.size() < RESERVOIR_SIZE) {
                        fReservoir.push_back({eventID, parentID, parentName, creatorProcess});
                    } else {
                        std::uniform_int_distribution<G4int> dist(0, fPhotonsSeen - 1);
                        G4int j = dist(fRng);
                        if (j < RESERVOIR_SIZE)
                            fReservoir[j] = {eventID, parentID, parentName, creatorProcess};
                    }
                }
            }
        }

        // ── Path tracking (only if enabled) ───────────────────
        if (gPathTrackingEnabled) {
            if (!gPhotonBuffers) gPhotonBuffers = new std::map<G4int, PhotonTrackBuffer>();
            if (!gLostBuffers)   gLostBuffers   = new std::map<G4int, PhotonTrackBuffer>();
            if (!gLostCount)     gLostCount     = new G4int(0);
            if (!gHandledTracks) gHandledTracks  = new std::set<G4int>();

            G4ThreeVector pos = step->GetPostStepPoint()->GetPosition();
            G4double      t   = step->GetPostStepPoint()->GetGlobalTime();
            // On the very first step, also record the creation vertex (pre-step = birth position)
            if (track->GetCurrentStepNumber() == 1) {
                G4ThreeVector birthPos = step->GetPreStepPoint()->GetPosition();
                G4double      birthT   = step->GetPreStepPoint()->GetGlobalTime();
                (*gPhotonBuffers)[trackID].AddStep(birthPos.x(), birthPos.y(), birthPos.z(), birthT);
                if (*gLostCount < MAX_LOST)
                    (*gLostBuffers)[trackID].AddStep(birthPos.x(), birthPos.y(), birthPos.z(), birthT);
            }

            (*gPhotonBuffers)[trackID].AddStep(pos.x(), pos.y(), pos.z(), t);
            if (*gLostCount < MAX_LOST)
                (*gLostBuffers)[trackID].AddStep(pos.x(), pos.y(), pos.z(), t);
        }

        // ── PMT detection ──────────────────────────────────────
        G4VPhysicalVolume* postVol = step->GetPostStepPoint()->GetPhysicalVolume();
        if (postVol == nullptr) {
            if (gPathTrackingEnabled) {
                if (*gLostCount < MAX_LOST) {
                    (*gLostBuffers)[trackID].FlushToNtuple(NTUPLE_LOST, trackID, eventID);
                    (*gLostCount)++;
                }
                gHandledTracks->insert(trackID);
                gPhotonBuffers->erase(trackID);
                gLostBuffers->erase(trackID);
            }
            return;
        }

        G4String volName = postVol->GetName();

        if (volName == "PMT_cathode") {
            // Record hit (Ntuple 1)
            am->FillNtupleDColumn(NTUPLE_HITS, 0, step->GetPostStepPoint()->GetPosition().x());
            am->FillNtupleDColumn(NTUPLE_HITS, 1, step->GetPostStepPoint()->GetPosition().y());
            am->FillNtupleDColumn(NTUPLE_HITS, 2, track->GetKineticEnergy());
            am->FillNtupleDColumn(NTUPLE_HITS, 3, track->GetLocalTime());
            am->FillNtupleIColumn(NTUPLE_HITS, 4, eventID);
            am->AddNtupleRow(NTUPLE_HITS);

            if (gPathTrackingEnabled) {
                (*gPhotonBuffers)[trackID].FlushToNtuple(NTUPLE_PATHS, trackID, eventID);
                gHandledTracks->insert(trackID);
                gPhotonBuffers->erase(trackID);
                gLostBuffers->erase(trackID);
            }
            track->SetTrackStatus(fStopAndKill);
        }

        if (volName == "PMT_backplate") {
            if (gPathTrackingEnabled) {
                if (*gLostCount < MAX_LOST) {
                    (*gLostBuffers)[trackID].FlushToNtuple(NTUPLE_LOST, trackID, eventID);
                    (*gLostCount)++;
                }
                gHandledTracks->insert(trackID);
                gPhotonBuffers->erase(trackID);
                gLostBuffers->erase(trackID);
            }
            track->SetTrackStatus(fStopAndKill);
        }
    }
};

// ================================================================
// TRACKING ACTION
// Only active when gParentageEnabled — maintains the particle name
// maps used for photon parentage lookup.
// ================================================================
class TrackingAction : public G4UserTrackingAction
{
public:
    void PreUserTrackingAction(const G4Track* track) override {
        if (!gParentageEnabled) return;

        if (!gTrackParticleMap)
            gTrackParticleMap = new std::map<G4int, G4String>();
        (*gTrackParticleMap)[track->GetTrackID()] =
            track->GetDefinition()->GetParticleName();

        // cache parent name for optical photons at birth
        if (track->GetDefinition() == G4OpticalPhoton::OpticalPhoton()) {
            if (!gPhotonParentNameMap)
                gPhotonParentNameMap = new std::map<G4int, G4String>();

            G4int parentID = track->GetParentID();
            G4String parentName = "unknown";
            if (gTrackParticleMap) {
                auto it = gTrackParticleMap->find(parentID);
                if (it != gTrackParticleMap->end())
                    parentName = it->second;
            }
            (*gPhotonParentNameMap)[track->GetTrackID()] = parentName;
        }
    }

    void PostUserTrackingAction(const G4Track* track) override {
        G4int tid = track->GetTrackID();

        if (gParentageEnabled) {
            if (gPhotonParentNameMap) gPhotonParentNameMap->erase(tid);
        }

        // Path tracking cleanup (independent of parentage)
        if (gPathTrackingEnabled &&
            track->GetDefinition() == G4OpticalPhoton::OpticalPhoton()) {

            G4bool handledByUs = gHandledTracks && gHandledTracks->count(tid) > 0;

            if (!handledByUs) {
                if (gLostCount && *gLostCount < MAX_LOST && gLostBuffers) {
                    auto it = gLostBuffers->find(tid);
                    if (it != gLostBuffers->end()) {
                        G4int eventID = G4RunManager::GetRunManager()
                                          ->GetCurrentEvent()->GetEventID();
                        it->second.FlushToNtuple(NTUPLE_LOST, tid, eventID);
                        (*gLostCount)++;
                    }
                }
            }

            if (gPhotonBuffers) gPhotonBuffers->erase(tid);
            if (gLostBuffers)   gLostBuffers->erase(tid);
            if (gHandledTracks) gHandledTracks->erase(tid);
        }
    }
};

// ================================================================
// EVENT ACTION
// ================================================================
class EventAction : public G4UserEventAction
{
private:
    SteppingAction* fSteppingAction;
public:
    EventAction(SteppingAction* sa) : fSteppingAction(sa) {}

    void BeginOfEventAction(const G4Event*) override {
        fSteppingAction->Reset();

        if (gParentageEnabled) {
            if (gTrackParticleMap)    gTrackParticleMap->clear();
            if (gPhotonParentNameMap) gPhotonParentNameMap->clear();
        }

        if (gPathTrackingEnabled) {
            if (gLostCount)     *gLostCount = 0;
            if (gHandledTracks)  gHandledTracks->clear();
        }
    }

    void EndOfEventAction(const G4Event* event) override {
        fSteppingAction->FlushParentage(); 

        auto am = G4AnalysisManager::Instance();
        am->FillNtupleIColumn(NTUPLE_PRODUCED, 0, fSteppingAction->GetPhotonsProduced());
        am->FillNtupleIColumn(NTUPLE_PRODUCED, 1, event->GetEventID());
        am->AddNtupleRow(NTUPLE_PRODUCED);
    }
};

// ================================================================
// RUN ACTION
// ================================================================
class RunAction : public G4UserRunAction
{
public:
    RunAction() {
        auto am = G4AnalysisManager::Instance();
        am->SetNtupleMerging(true);

        // 0: MuonEntry
        am->CreateNtuple("MuonEntry", "Muon Entry Points");
        am->CreateNtupleDColumn("y_entry");
        am->CreateNtupleDColumn("z_entry");
        am->CreateNtupleIColumn("EvID");
        am->FinishNtuple();

        // 1: PhotonHits
        am->CreateNtuple("PhotonHits", "Optical Photon PMT Hits");
        am->CreateNtupleDColumn("x_hit");
        am->CreateNtupleDColumn("y_hit");
        am->CreateNtupleDColumn("energy");
        am->CreateNtupleDColumn("arrival_time");
        am->CreateNtupleIColumn("EvID");
        am->FinishNtuple();

        // 2: PhotonsProduced
        am->CreateNtuple("PhotonsProduced", "Total Scintillation Photons");
        am->CreateNtupleIColumn("n_produced");
        am->CreateNtupleIColumn("EvID");
        am->FinishNtuple();

        // 3: PhotonParentage
        am->CreateNtuple("PhotonParentage", "Scintillation Photon Origins");
        am->CreateNtupleIColumn("EvID");
        am->CreateNtupleIColumn("ParentID");
        am->CreateNtupleSColumn("ParentParticle");
        am->CreateNtupleSColumn("CreatorProcess");
        am->FinishNtuple();

        // 4 & 5: Path ntuples — always created so ROOT file structure is
        // consistent regardless of toggle. They will just be empty when
        // path tracking is disabled.
        am->CreateNtuple("PhotonPaths", "Detected Photon Track Positions");
        am->CreateNtupleDColumn("x");
        am->CreateNtupleDColumn("y");
        am->CreateNtupleDColumn("z");
        am->CreateNtupleDColumn("t");
        am->CreateNtupleIColumn("TrackID");
        am->CreateNtupleIColumn("EvID");
        am->FinishNtuple();

        am->CreateNtuple("LostPhotonPaths", "Lost Photon Track Positions");
        am->CreateNtupleDColumn("x");
        am->CreateNtupleDColumn("y");
        am->CreateNtupleDColumn("z");
        am->CreateNtupleDColumn("t");
        am->CreateNtupleIColumn("TrackID");
        am->CreateNtupleIColumn("EvID");
        am->FinishNtuple();
    }

    void BeginOfRunAction(const G4Run*) override {
        G4AnalysisManager::Instance()->OpenFile("scoring.root");
    }

    void EndOfRunAction(const G4Run*) override {
        auto am = G4AnalysisManager::Instance();
        am->Write();
        am->CloseFile();
    }
};

// ================================================================
// ACTION INITIALIZATION
// ================================================================
class Action : public G4VUserActionInitialization
{
public:
    void BuildForMaster() const override {
        SetUserAction(new RunAction);
    }

    void Build() const override {
        SetUserAction(new Generator);
        SetUserAction(new RunAction);
        SteppingAction* sa = new SteppingAction();
        SetUserAction(sa);
        SetUserAction(new EventAction(sa));
        SetUserAction(new TrackingAction);   // always registered
    }
};

// ================================================================
// MAIN
// ================================================================
int main(int argc, char** argv)
{
    // ── Read environment toggles ──────────────────────────────────────────────
    const char* envTrack     = std::getenv("ENABLE_PATH_TRACKING");
    const char* envParentage = std::getenv("ENABLE_PARENTAGE");
    const char* envThreads   = std::getenv("G4_NUM_THREADS");

    gPathTrackingEnabled = (envTrack     && std::string(envTrack)     == "1");
    gParentageEnabled    = (envParentage && std::string(envParentage) == "1");

    G4cout << ">>> Path tracking : "
           << (gPathTrackingEnabled ? "ENABLED" : "DISABLED") << G4endl;
    G4cout << ">>> Parentage     : "
           << (gParentageEnabled    ? "ENABLED" : "DISABLED (production mode)") << G4endl;

    auto run = G4RunManagerFactory::CreateRunManager();

    // Set thread count from environment (default: all available cores)
    if (envThreads) {
        int nThreads = std::stoi(envThreads);
        run->SetNumberOfThreads(nThreads);
        G4cout << ">>> Threads       : " << nThreads << G4endl;
    }

    G4PhysListFactory f;
    G4VModularPhysicsList* physicsList = f.GetReferencePhysList("QGSP_BIC_EMZ");
    physicsList->RegisterPhysics(new G4OpticalPhysics());
    run->SetUserInitialization(physicsList);

    run->SetUserInitialization(new Detector);
    run->SetUserInitialization(new Action);

    // Visualisation only in interactive mode
    G4UIExecutive* ui = nullptr;
    if (argc == 1) {
        ui = new G4UIExecutive(argc, argv);
        auto vis = new G4VisExecutive("quiet");
        vis->Initialize();
        ui->SessionStart();
        delete ui;
        delete vis;
    } else {
        G4String cmd = "/control/execute ", macroFile = argv[1];
        G4UImanager::GetUIpointer()->ApplyCommand(cmd + macroFile);
    }

    delete run;
}
// -*- C++; indent-tabs-mode:nil; tab-width:2 -*-
// vim: ft=cpp:ts=2:sts=2:sw=2:et
