//
//  KRMLPNet.h
//  KRMLP
//
//  Created by Kalvar Lin on 2016/5/2.
//  Copyright © 2016年 Kalvar Lin. All rights reserved.
//

@import Foundation;
#import "KRMLPOptimization.h"

typedef NS_ENUM(NSInteger, KRMLPNetActivations)
{
    KRMLPNetActivationDefault = 0,  // Default is Sigmoid.
    KRMLPNetActivationSigmoid,      // S 型曲線, [0.0, 1.0]
    KRMLPNetActivationTanh,         // 雙 S 曲線, [-1.0, 1.0]
    KRMLPNetActivationRBF,          // RBF 分佈曲線, [0.0, 1.0]
    KRMLPNetActivationLinear,       // 線性函數, (0, 1)
    KRMLPNetActivationCubic         // 三次函數, [-oo, oo]
};

@interface KRMLPNet : NSObject <NSCopying, NSCoding>

@property (nonatomic, strong) NSNumber *indexKey;                    // 第幾筆資料或第幾顆神經元
@property (nonatomic, strong) NSMutableArray <NSNumber *> *weights;  // 連結至本 Net 的所有權重值
@property (nonatomic, strong) NSMutableArray <NSNumber *> *features; // Features are dynamic changing in every moment, the keep in weak reference is enough in MLP structure without deep copying.
@property (nonatomic, assign) double bias;                           // 偏權值, default is 1.0f
@property (nonatomic, assign) double outputValue;                    // 神經元輸出值
@property (nonatomic, readonly) double outputError;                  // 神經元輸出誤差值
@property (nonatomic, assign) double deltaValue;                     // 神經元修正誤差值(Delta), 用於修正權重、偏權值
@property (nonatomic, strong) NSMutableArray <NSNumber *> *lastDeltaWeights; // 上一次的權重修正量
@property (nonatomic, strong) NSMutableArray <NSNumber *> *deltaWeights;     // 當前的權重修正量
@property (nonatomic, strong) NSMutableArray <NSNumber *> *lastGradients;    // 上一次每條權重的梯度下降量
@property (nonatomic, strong) NSMutableArray <NSNumber *> *gradients;        // 當前每條權重的梯度下降量
@property (nonatomic, assign) NSInteger updatedTimes;                // 神經元更新過的次數
@property (nonatomic, assign) double slope;                          // 活化函數曲線的坡度, ex : sigmoid 的 alpha value 會決定該曲線是平滑或陡峭
@property (nonatomic, assign) KRMLPNetActivations netActivation;     // The activation of net, if it is not KRMLPNetActivationDefault that means this net already set the active function.
@property (nonatomic, readonly) double outputPartialDerivative;      // f'(outputValue) the partial differential of output value.
@property (nonatomic, strong) KRMLPOptimization *optimization;       // 權重優化方法

- (instancetype)init;

- (void)addWeightsFromArray:(NSArray <NSNumber *> *)theWeights;
- (void)addWeight:(NSNumber *)theWeights;
- (void)addFeature:(NSNumber *)aFeature;
- (void)addFeaturesFromArray:(NSArray <NSNumber *> *)theFeatures;
- (void)removeWeights;
- (void)removeFeatures;
- (void)renewWeights:(NSArray <NSNumber *> *)newWeights;

- (void)setActivationIfNeeded:(KRMLPNetActivations)activeFunction;
- (double)outputWithInputs:(NSArray <NSNumber *> *)inputs;

- (void)randomizeWeightsAtCount:(NSInteger)randomCount max:(double)max min:(double)min;
- (void)randomizeWeightsAtMax:(double)max min:(double)min;

- (double)newBiasWithLearningRate:(double)learningRate;
- (NSArray <NSNumber *> *)newWeightsWithLayerOutputs:(NSArray <NSNumber *> *)layerOutputs learningRate:(double)learningRate;

@end

@interface KRMLPNet (NSCoding)

- (void)encodeObject:(id)object forKey:(NSString *)key;
- (id)decodeForKey:(NSString *)key;

@end