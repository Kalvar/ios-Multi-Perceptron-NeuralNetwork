//
//  KRMLPOutputLayer.h
//  KRMLP
//
//  Created by Kalvar Lin on 2016/6/19.
//  Copyright © 2016年 Kalvar Lin. All rights reserved.
//

#import "KRMLPLayer.h"
#import "KRMLPOutputNet.h"

@interface KRMLPOutputLayer : KRMLPLayer

@property (nonatomic, strong) NSMutableArray <KRMLPOutputNet *> *nets;
@property (nonatomic, strong) NSMutableArray <NSNumber *> *outputs;
@property (nonatomic, assign) KRMLPNetActivations activeFunction; // All of nets are using specific active function.

- (void)addNet:(KRMLPOutputNet *)net;
- (void)addNetsFromArray:(NSArray <KRMLPOutputNet *> *)layerNets;
- (NSArray <NSNumber *> *)directOutputsWithInputs:(NSArray <NSNumber *> *)inputs;
- (NSArray <NSNumber *> *)layerOutputsWithInputs:(NSArray <NSNumber *> *)inputs;
- (double)calculateCostAndDeltaWithTargets:(NSArray <NSNumber *> *)targets;
- (void)updateWithHiddenLayerOutputs:(NSArray <NSNumber *> *)layerOutputs learningRate:(double)learningRate;

@end
