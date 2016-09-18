//
//  KRMLP.h
//  KRMLP
//
//  Created by Kalvar Lin on 2016/4/27.
//  Copyright © 2016年 Kalvar Lin. All rights reserved.
//

#import <Foundation/Foundation.h>

#import "KRMLPPattern.h"
#import "KRMLPHiddenNet.h"
#import "KRMLPOutputNet.h"
#import "KRMLPInputLayer.h"
#import "KRMLPHiddenLayer.h"
#import "KRMLPOutputLayer.h"
#import "KRMLPNetworkOutput.h"
#import "KRMLPTrainingOutput.h"
#import "KRMLPOptimization.h"

@class KRMLP;

typedef void(^KRMLPNetworkOutputBlock)(KRMLP *network);                         // 網路最後的輸出結果 Block
typedef void(^KRMLPIterationBlock)(KRMLP *network);                             // 每一個迭代的 Block
typedef void(^KRMLPTrainingOutputBlock)(KRMLPTrainingOutput *trainingOutput);   // 訓練時的輸出 Block
typedef void(^KRMLPNetworkPredicationBlock)(KRMLPNetworkOutput *networkOutput); // 預測的輸出 Block

@interface KRMLP : NSObject

@property (nonatomic, strong) KRMLPInputLayer *inputLayer; // InputLayer controls patterns, the every net is a pattern.
@property (nonatomic, strong) NSMutableArray <KRMLPHiddenLayer *> *hiddenLayers;
@property (nonatomic, strong) KRMLPOutputLayer *outputLayer;

@property (nonatomic, readwrite) NSMutableArray <KRMLPPattern *> *patterns; // The patterns belongs to [inputLayer nets].
@property (nonatomic, readonly) NSInteger patternsCount;      // Counting how many patterns.
@property (nonatomic, readonly) NSInteger hiddenLayerCount;   // Counting how many hidden layers.
@property (nonatomic, readonly) NSInteger outputNetsCount;    // Counting how many output nets.
@property (nonatomic, readonly) NSInteger inputNetsCount;     // Counting how many input nets.
@property (nonatomic, assign) BOOL initialOptimize;           // To optimize the initial weights.

@property (nonatomic, assign) KRMLPNetActivations networkActivation; // The used active function of whole network.

@property (nonatomic, assign) double learningRate; // Default is 0.8

@property (nonatomic, assign) double initialMaxWeight; // Default random max weight-value
@property (nonatomic, assign) double initialMinWeight; // Default random min weight-value

@property (nonatomic, assign) double convergenceError; // Normally between 10^-3 and 10^-6
@property (nonatomic, assign) NSInteger maxIteration;  // The max iteration number of limitation.
@property (nonatomic, readonly) NSInteger iteration;   // The current iteration of running.

+ (instancetype)sharedNetwork;
- (instancetype)init;

- (NSInteger)calculateLayerNetCountWithInputCount:(NSInteger)inputCount outputCount:(NSInteger)outputCount;

- (KRMLPPattern *)createPatternWithFeatures:(NSArray <NSNumber *> *)features targets:(NSArray <NSNumber *> *)targets;
- (KRMLPHiddenLayer *)createHiddenLayerWithIndex:(NSNumber *)layerIndex nets:(NSArray <KRMLPHiddenNet *> *)nets;
- (KRMLPHiddenLayer *)createHiddenLayerWithNetCount:(NSInteger)netCount inputCount:(NSInteger)inputCount maxWeight:(double)maxWeight min:(double)minWeight;
- (KRMLPHiddenLayer *)createHiddenLayerWithNetCount:(NSInteger)netCount inputCount:(NSInteger)inputCount;
- (KRMLPHiddenLayer *)createHiddenLayerWithAutomaticSetting;
- (KRMLPHiddenLayer *)createHiddenLayerDependsOnHiddenLayer:(KRMLPHiddenLayer *)dependentLayer netCount:(NSInteger)netCount maxWeight:(double)maxWeight min:(double)minWeight;
- (KRMLPHiddenLayer *)createHiddenLayerDependsOnHiddenLayer:(KRMLPHiddenLayer *)dependentLayer netCount:(NSInteger)netCount;

- (void)addPattern:(KRMLPPattern *)pattern;
- (void)addPatternsFromArray:(NSArray <KRMLPPattern *> *)samples;
- (void)addHiddenLayer:(KRMLPHiddenLayer *)hiddenLayer;
- (void)addHiddenLayer:(KRMLPHiddenLayer *)hiddenLayer dependsOnHiddenLayer:(KRMLPHiddenLayer *)dependentLayer;
- (void)addHiddenNet:(KRMLPHiddenNet *)hiddenNet forLayerIndex:(NSInteger)layerIndex;
- (void)addOutpuNet:(KRMLPOutputNet *)outputNet;
- (void)setupOutputLayerWithNetMaxWeight:(double)maxWeight min:(double)minWeight;
- (void)setupOutputLayer;

- (void)trainingWithCompletion:(KRMLPNetworkOutputBlock)completion iteration:(KRMLPIterationBlock)iteration training:(KRMLPTrainingOutputBlock)training;
- (void)trainingWithCompletion:(KRMLPNetworkOutputBlock)completion iteration:(KRMLPIterationBlock)iteration;
- (void)training;
- (void)predicateWithFeatures:(NSArray <NSNumber *> *)features completion:(KRMLPNetworkPredicationBlock)completion;

- (void)saveForKey:(NSString *)key;
- (void)recoverForKey:(NSString *)key;

- (void)setupOptimizationMethod:(KRMLPOptimizationMethods)method inertialRate:(double)inertialRate;
- (void)setupOptimizationMethod:(KRMLPOptimizationMethods)method;
- (void)setupOptimizationInertialRate:(double)inertialRate;

@end
