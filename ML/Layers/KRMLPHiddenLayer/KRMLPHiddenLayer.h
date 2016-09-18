//
//  KRMLPHiddenLayer.h
//  KRMLP
//
//  Created by Kalvar Lin on 2016/6/19.
//  Copyright © 2016年 Kalvar Lin. All rights reserved.
//

#import "KRMLPLayer.h"
#import "KRMLPHiddenNet.h"

@interface KRMLPHiddenLayer : KRMLPLayer

@property (nonatomic, strong) NSNumber *index; // 是第幾層的 Hidden Layer
@property (nonatomic, strong) NSMutableArray <KRMLPHiddenNet *> *nets;
@property (nonatomic, strong) NSMutableArray *outputs; // Layer outputs
@property (nonatomic, assign) KRMLPNetActivations activeFunction; // All of nets are using specific active function.

- (instancetype)initWithIndex:(NSNumber *)layerIndex nets:(NSArray <KRMLPHiddenNet *> *)layerNets;
- (instancetype)init;

- (void)addNet:(KRMLPHiddenNet *)net;
- (void)addNetsFromArray:(NSArray <KRMLPHiddenNet *> *)layerNets;
- (NSArray <NSNumber *> *)layerOutputsWithInputs:(NSArray <NSNumber *> *)inputs;
- (NSArray <KRMLPNet *> *)calculateDeltasWithLastLayerDeltas:(NSArray <KRMLPNet *> *)lastNets;
- (void)updateWithLastLayerOutputs:(NSArray <NSNumber *> *)layerOutputs learningRate:(double)learningRate;

@end
