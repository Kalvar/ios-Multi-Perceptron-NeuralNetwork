//
//  KRRBFPassed.h
//  KRMLP
//
//  Created by Kalvar Lin on 2016/4/12.
//  Copyright © 2016年 Kalvar Lin. All rights reserved.
//

#import "KRMLPNet.h"

@class KRMLPHiddenLayer;
@class KRMLPOutputLayer;

// This saved the main parameters of trained network.
@interface KRMLPPassed : NSObject <NSCoding>

@property (nonatomic, strong) NSMutableArray <KRMLPHiddenLayer *> *hiddenLayers;
@property (nonatomic, strong) KRMLPOutputLayer *outputLayer;
@property (nonatomic, assign) KRMLPNetActivations networkActivation;
@property (nonatomic, assign) double learningRate;

- (instancetype)init;

@end
