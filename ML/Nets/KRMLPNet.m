//
//  KRMLPNet.m
//  KRMLP
//
//  Created by Kalvar Lin on 2016/5/2.
//  Copyright © 2016年 Kalvar Lin. All rights reserved.
//

#import "KRMLPNet.h"
#import "KRMLPActivation.h"
#import "KRMathLib.h"

@interface KRMLPNet ()

@property (nonatomic, strong) KRMLPActivation *activation;
@property (nonatomic, strong) KRMathLib *mathLib;

@property (nonatomic, weak) NSCoder *coder; // For NSCoding usage and child-class use in inherited this class.

@end

@implementation KRMLPNet (NSCoding)

- (void)encodeObject:(id)object forKey:(NSString *)key
{
    if( nil != object )
    {
        [self.coder encodeObject:object forKey:key];
    }
}

- (id)decodeForKey:(NSString *)key
{
    return [self.coder decodeObjectForKey:key];
}

@end

@implementation KRMLPNet (Weights)

- (void)recordLastDeltaWeightsFromArray:(NSArray *)deltas
{
    if( nil != deltas )
    {
        [self.lastDeltaWeights removeAllObjects];
        [self.lastDeltaWeights addObjectsFromArray:deltas];
    }
}

- (void)recordLastGradientsFromArray:(NSArray *)gradients
{
    if( nil != gradients )
    {
        [self.lastGradients removeAllObjects];
        [self.lastGradients addObjectsFromArray:gradients];
    }
}

@end

@implementation KRMLPNet (Activations)

// 偏微分, f'(x)
- (double)partialDifferential:(double)x slope:(double)slope
{
    double dashValue = 0.0f;
    switch (self.netActivation)
    {
        case KRMLPNetActivationDefault:
        case KRMLPNetActivationSigmoid:
            dashValue = [self.activation dashSigmoid:x slope:slope];
            break;
        case KRMLPNetActivationTanh:
            dashValue = [self.activation dashTanh:x slope:slope];
            break;
        case KRMLPNetActivationRBF:
            dashValue = [self.activation dashRBF:x sigma:slope];
            break;
        case KRMLPNetActivationLinear:
            dashValue = [self.activation dashSgn:x];
            break;
        case KRMLPNetActivationCubic:
            dashValue = [self.activation dashCubic:x];
            break;
        default:
            break;
    }
    return dashValue;
}

- (double)activate:(double)x slope:(double)slope
{
    double y = 0.0f;
    switch (self.netActivation)
    {
        case KRMLPNetActivationDefault:
        case KRMLPNetActivationSigmoid:
            y = [self.activation sigmoid:x slope:slope];
            break;
        case KRMLPNetActivationTanh:
            y = [self.activation tanh:x slope:slope];
            break;
        case KRMLPNetActivationRBF:
            y = [self.activation rbf:x sigma:slope];
            break;
        case KRMLPNetActivationLinear:
            y = [self.activation sgn:x];
            break;
        case KRMLPNetActivationCubic:
            y = [self.activation cubic:x];
            break;
        default:
            break;
    }
    return y;
}

@end

@implementation KRMLPNet

- (instancetype)init
{
    self = [super init];
    if( self )
    {
        _weights          = [NSMutableArray new];
        _features         = [NSMutableArray new];
        _bias             = 0.25f;
        _outputValue      = 0.0f;
        _deltaValue       = 0.0f;
        _lastDeltaWeights = [NSMutableArray new];
        _deltaWeights     = [NSMutableArray new];
        _lastGradients    = [NSMutableArray new];
        _gradients        = [NSMutableArray new];
        _updatedTimes     = 0;
        _slope            = 1.0f;
        
        _netActivation    = KRMLPNetActivationDefault;
        _activation       = [[KRMLPActivation alloc] init];
        _mathLib          = [[KRMathLib alloc] init];
        _optimization     = [KRMLPOptimization shared];
    }
    return self;
}

- (void)addWeightsFromArray:(NSArray <NSNumber *> *)theWeights
{
    if( nil != theWeights && [theWeights count] > 0 )
    {
        [_weights addObjectsFromArray:theWeights];
    }
}

- (void)addWeight:(NSNumber *)theWeights
{
    if( nil != theWeights )
    {
        [_weights addObject:theWeights];
    }
}

- (void)addFeature:(NSNumber *)aFeature
{
    if( nil != aFeature )
    {
        [_features addObject:aFeature];
    }
}

- (void)addFeaturesFromArray:(NSArray <NSNumber *> *)theFeatures
{
    if( nil != theFeatures && [theFeatures count] > 0 )
    {
        [_features addObjectsFromArray:theFeatures];
    }
}

- (void)removeWeights
{
    [_weights removeAllObjects];
}

- (void)removeFeatures
{
    [_features removeAllObjects];
}

- (void)renewWeights:(NSArray <NSNumber *> *)newWeights
{
    if( nil != newWeights )
    {
        _updatedTimes += 1;
        [self removeWeights];
        [self addWeightsFromArray:newWeights];
    }
}

// If netActivation is default value that we need to set it to specific activeFunction.
- (void)setActivationIfNeeded:(KRMLPNetActivations)activeFunction
{
    if( _netActivation == KRMLPNetActivationDefault )
    {
        _netActivation = activeFunction;
    }
}

// Net output.
- (double)outputWithInputs:(NSArray<NSNumber *> *)inputs
{
    double sumValue = [_mathLib sumMatrix:inputs anotherMatrix:_weights];
    _outputValue    = [self activate:sumValue slope:_slope];
    return _outputValue;
}

#pragma mark - Randoms
- (void)randomizeWeightsAtCount:(NSInteger)randomCount max:(double)max min:(double)min
{
    [_weights removeAllObjects];
    for( NSInteger i=0; i<randomCount; i++ )
    {
        [_weights addObject:@([_mathLib randomDoubleMax:max min:min])];
    }
}

// 隨機設定 randomCount 個權重
- (void)randomizeWeightsAtMax:(double)max min:(double)min
{
    [self randomizeWeightsAtCount:[_features count] max:max min:min];
}

#pragma mark - Updating
- (double)newBiasWithLearningRate:(double)learningRate
{
    // Formula of Bias : new b(j) = -L * Error(k) + b(j)
    return ((-learningRate) * _deltaValue) + _bias;
}

- (NSArray <NSNumber *> *)newWeightsWithLayerOutputs:(NSArray <NSNumber *> *)layerOutputs learningRate:(double)learningRate
{
    // 先記錄上一次的修正量 (在初次更新權重的情況下，_deltaWeights 會是空陣列，故上一次的修正量也會是空記錄)
    [self recordLastDeltaWeightsFromArray:_deltaWeights];
    [self recordLastGradientsFromArray:_gradients];
    [_deltaWeights removeAllObjects];
    [_gradients removeAllObjects];
    
    __weak typeof(self) weakSelf = self;
    __block NSMutableArray <NSNumber *> *newWeights = [NSMutableArray new];
    [_weights enumerateObjectsUsingBlock:^(NSNumber * _Nonnull weight, NSUInteger idx, BOOL * _Nonnull stop) {
        __strong typeof(weakSelf) strongSelf = weakSelf;
        // Formula of Weight :
        //   new w(jk)       = -L * delta value * Output(j) + w(jk)
        //                   = -L * ( d(j) - y(j) ) * -f'( y(j) ) * Output(j) + w(jk)
        //   負號相消, 得最終式 = L * Error(k) * Output(j) + w(jk), 其中 Error(k) = delta values
        
        // 先取出該權重對應的上一層 Hidden Layer / Input Layer Net Output
        double mappedOutput = [[layerOutputs objectAtIndex:idx] doubleValue];
        
        // 記錄當前的權重梯度下降量
        double gradientWeight = -(strongSelf.deltaValue * mappedOutput);
        [strongSelf.gradients addObject:@(gradientWeight)];
        
        // 運算權重修正量
        double deltaWeight  = [strongSelf.optimization deltaWeightAtIndex:idx net:strongSelf mappedOutput:mappedOutput learningRate:learningRate];
        double newWeight    = deltaWeight + [weight doubleValue];
        [newWeights addObject:@(newWeight)];
        
        // 記錄當前的權重修正量
        [strongSelf.deltaWeights addObject:@(deltaWeight)];
    }];
    
    return newWeights;
}

#pragma mark - Getter
- (NSNumber *)indexKey
{
    if( nil == _indexKey )
    {
        // To use milliseconds to be default indexKey if it is nil.
        _indexKey = @([[NSDate date] timeIntervalSince1970] * 1000);
    }
    return _indexKey;
}

// 對 Net 的輸出值作偏微分, f'(outputValue)
- (double)outputPartialDerivative
{
    return [self partialDifferential:_outputValue slope:_slope];
}

#pragma mark - NSCopying
- (instancetype)copyWithZone:(NSZone *)zone
{
    KRMLPNet *_net = [[KRMLPNet alloc] init];
    _net.weights   = [[NSMutableArray alloc] initWithArray:_weights copyItems:YES]; // Deeply copying
    _net.indexKey  = [_indexKey copy];
    _net.bias      = _bias;
    return _net;
}

#pragma mark - NSCoding
- (void)encodeWithCoder:(NSCoder *)aCoder
{
    self.coder = aCoder;
    [self encodeObject:self.indexKey forKey:@"indexKey"];
    [self encodeObject:_weights forKey:@"weights"];
    [self encodeObject:@(_bias) forKey:@"bias"];
    [self encodeObject:@(_slope) forKey:@"slope"];
    [self encodeObject:@(_netActivation) forKey:@"netActivation"];
    
    [self encodeObject:@(_optimization.method) forKey:@"optimization.method"];
    [self encodeObject:@(_optimization.inertialRate) forKey:@"optimization.inertialRate"];
}

- (instancetype)initWithCoder:(NSCoder *)aDecoder
{
    self = [super init];
    if(self)
    {
        self.coder        = aDecoder;
        _indexKey         = [self decodeForKey:@"indexKey"];
        _weights          = [self decodeForKey:@"weights"];
        _bias             = [[self decodeForKey:@"bias"] doubleValue];
        _slope            = [[self decodeForKey:@"slope"] doubleValue];
        _netActivation    = [[self decodeForKey:@"netActivation"] integerValue];
        
        // To initialize others must-have parameters but we didn't save.
        _activation       = [[KRMLPActivation alloc] init];
        _mathLib          = [[KRMathLib alloc] init];
        _outputValue      = 0.0f;
        _deltaValue       = 0.0f;
        _updatedTimes     = 0;
        _lastDeltaWeights = [NSMutableArray new];
        _deltaWeights     = [NSMutableArray new];
        _lastGradients    = [NSMutableArray new];
        _gradients        = [NSMutableArray new];
        
        _optimization              = [KRMLPOptimization shared];
        _optimization.method       = [[self decodeForKey:@"optimization.method"] integerValue];
        _optimization.inertialRate = [[self decodeForKey:@"optimization.inertialRate"] doubleValue];
    }
    return self;
}

@end