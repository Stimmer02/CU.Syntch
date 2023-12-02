#include "DynamicsController.h"


using namespace synthesizer;

void DynamicsController::calculateDynamicsProfile(settings& settings){
    noteDynamicsProfile.clear();

    for (uint i = 0; i < settings.attack.duration; i++){
        noteDynamicsProfile.push_back(i/double(settings.attack.duration));
    }
    for (uint i = 0; i < settings.sustain.duration; i++){
        noteDynamicsProfile.push_back(1);
    }
    for (uint i = 0; i < settings.fade.duration; i++){
        noteDynamicsProfile.push_back((1-i/double(settings.fade.duration)) * (1-settings.fadeTo) + settings.fadeTo);
    }

    settings.dynamicsDuration = noteDynamicsProfile.size();
    if (settings.fade.duration == 0){
        settings.fadeTo = 1;
    } else {
        settings.fadeTo = settings.rawFadeTo;
    }
}

void DynamicsController::calculateReleaseProfile(settings& settings){
    noteReleaseProfile.clear();

    for (uint i = 0; i < settings.release.duration; i++){
        noteReleaseProfile.push_back(1-i/double(settings.release.duration));
    }
}


//may turn usefull if dynamics function won't be linear
// void DynamicsController::createReleaseDynamicMap(){
//     if (releaseDynamicMap != nullptr){
//         delete[] releaseDynamicMap;
//     }
//     const ulong& size = noteReleaseProfile.size();
//     releaseDynamicMap = new uint[size];
//     for (uint i = 0, j = noteDynamicsProfile.size() - 1; i < size; i++){
//         double lastDifference;
//         double difference = std::abs(noteDynamicsProfile.at(j) - noteReleaseProfile.at(i));
//         do {
//             lastDifference = difference;
//             j--;
//             difference = std::abs(noteDynamicsProfile.at(j) - noteReleaseProfile.at(i));
//         } while (lastDifference <= difference && j > 0);
//         releaseDynamicMap[i] = j;
//     }
// }

const double* DynamicsController::getDynamicsProfile(){
    return noteDynamicsProfile.data();
}

const double* DynamicsController::getReleaseProfile(){
    return noteReleaseProfile.data();
}


uint DynamicsController::getDynamicsProfileLength(){
    return noteDynamicsProfile.size();
}

uint DynamicsController::getReleaseProfileLength(){
    return noteReleaseProfile.size();
}
