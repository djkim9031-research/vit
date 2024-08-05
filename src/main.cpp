#include <iostream>
#include "vit.h"

int main(){
    ViT_trainer("../vit_config.yaml", "../data/", "../data/model_weights", "", 25600, 1280);

    return 0;
}