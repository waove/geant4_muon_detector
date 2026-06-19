// ==============================================================
// main.cc — CosmicMuonSim entry point
//
// Usage:
//   ./CosmicMuonSim              → interactive session
//   ./CosmicMuonSim run.mac      → batch mode
//   ./CosmicMuonSim -n 50000     → batch, 50k events, no macro
// ==============================================================

#include "DetectorConstruction.hh"
#include "ActionInitialization.hh"

#include "G4RunManagerFactory.hh"
#include "G4UImanager.hh"
#include "G4UIExecutive.hh"
#include "G4VisExecutive.hh"
#include "G4PhysListFactory.hh"
#include "G4StepLimiterPhysics.hh"

#include "Randomize.hh"
#include <chrono>

#include <string>

int main(int argc, char** argv)
{

    long seed = std::chrono::system_clock::now().time_since_epoch().count();
    G4Random::setTheSeed(seed);
    G4cout << "Random seed: " << seed << G4endl;

    // ---- Parse simple CLI -------------------------------------------
    G4String macroFile = "";
    G4int    nEvents   = 0;

    for (int i = 1; i < argc; ++i) {
        G4String arg(argv[i]);
        if (arg == "-n" && i + 1 < argc) {
            nEvents = std::stoi(argv[++i]);
        } else {
            macroFile = arg;
        }
    }

    // ---- Run manager ------------------------------------------------
    // Default = MT if Geant4 was compiled with MT support.
    // Each worker thread writes its own output files; the master
    // merges them in EndOfRunAction.
    auto* runManager = G4RunManagerFactory::CreateRunManager(
                           G4RunManagerType::Serial);

    // ---- Mandatory user classes -------------------------------------
    runManager->SetUserInitialization(new DetectorConstruction());

    G4PhysListFactory f;
    auto* physicsList = f.GetReferencePhysList("FTFP_BERT_EMZ");
    physicsList->RegisterPhysics(new G4StepLimiterPhysics());
    runManager->SetUserInitialization(physicsList);

    runManager->SetUserInitialization(new ActionInitialization());

    // ---- Vis & UI ---------------------------------------------------
    auto* visManager = new G4VisExecutive;
    visManager->Initialize();
    auto* UImanager = G4UImanager::GetUIpointer();

    if (macroFile.size()) {
        // Batch mode — execute macro
        UImanager->ApplyCommand("/control/execute " + macroFile);
    }
    else if (nEvents > 0) {
        // Quick batch: initialise + run N events
        UImanager->ApplyCommand("/run/initialize");
        UImanager->ApplyCommand("/run/beamOn " + std::to_string(nEvents));
    }
    else {
        // Interactive
        auto* ui = new G4UIExecutive(argc, argv);
        UImanager->ApplyCommand("/control/execute vis.mac");
        ui->SessionStart();
        delete ui;
    }

    delete visManager;
    delete runManager;
    return 0;
}
