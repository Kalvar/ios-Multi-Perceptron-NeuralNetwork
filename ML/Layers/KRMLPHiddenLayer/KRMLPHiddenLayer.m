//
//  KRMLPHiddenLayer.m
//  KRMLP
//
//  Created by Kalvar Lin on 2016/6/19.
//  Copyright © 2016年 Kalvar Lin. All rights reserved.
//

#import "KRMLPHiddenLayer.h"
#import "KRMLPHiddenNet.h"
#import "KRMathLib.h"

@interface KRMLPHiddenLayer ()

@property (nonatomic, strong) KRMathLib *mathLib;

@end

@implementation KRMLPHiddenLayer

- (instancetype)initWithIndex:(NSNumber *)layerIndex nets:(NSArray <KRMLPHiddenNet *> *)layerNets
{
    self = [super init];
    if( self )
    {
        _index   = layerIndex;
        _nets    = [NSMutableArray new];
        [self addNetsFromArray:layerNets];
        _outputs = [NSMutableArray new];
        _mathLib = [[KRMathLib alloc] init];
    }
    return self;
}

- (instancetype)init
{
    return [self initWithIndex:0 nets:nil];
}

- (void)addNet:(KRMLPHiddenNet *)net
{
    if( net )
    {
        [_nets addObject:net];
    }
}

- (void)addNetsFromArray:(NSArray<KRMLPHiddenNet *> *)layerNets
{
    if( layerNets )
    {
        [_nets addObjectsFromArray:layerNets];
    }
}

// To calculate the layer outputs.
- (NSArray <NSNumber *> *)layerOutputsWithInputs:(NSArray <NSNumber *> *)inputs
{
    [_outputs removeAllObjects];
    __weak typeof(self) weakSelf = self;
    [_nets enumerateObjectsUsingBlock:^(KRMLPHiddenNet * _Nonnull net, NSUInteger idx, BOOL * _Nonnull stop) {
        __strong typeof(weakSelf) strongSelf = weakSelf;
        [strongSelf.outputs addObject:@([net outputWithInputs:inputs])];
    }];
    return _outputs;
}

- (NSArray <KRMLPNet *> *)calculateDeltasWithLastLayerDeltas:(NSArray <KRMLPNet *> *)lastNets
{
    // Formula : SUM(error(k) * w(jk)) * f'(netOutput), the error(k) is delta(k).
    [_nets enumerateObjectsUsingBlock:^(KRMLPHiddenNet * _Nonnull hiddenNet, NSUInteger netIndex, BOOL * _Nonnull stop) {
        __block double deltaValue = 0.0f;
        // 上一層的 Nets
        [lastNets enumerateObjectsUsingBlock:^(KRMLPNet * _Nonnull lastNet, NSUInteger lastIndex, BOOL * _Nonnull stop) {
            // 取出該 Net 的 Delta Value (誤差修正量) 來計算隱藏層的權重和偏權修正量
            // SUM(對應到此 Hidden Net 的所有權重值各自乘上對應的上一層 Nets 的 Delta Value)
            double netWeight  = [[lastNet.weights objectAtIndex:netIndex] doubleValue];
            // Deltak is error(k).
            deltaValue       += lastNet.deltaValue * netWeight;
        }];
        // Delta-value of current net = SUM(delta(k) * w(jk)) * f'(netOutput)
        deltaValue *= hiddenNet.outputPartialDerivative;
        // The error(k) of next hidden layer.
        hiddenNet.deltaValue = deltaValue;
    }];
    return _nets;
}

// Updating the weights and biases with hidden layer or input layer outputs.
- (void)updateWithLastLayerOutputs:(NSArray <NSNumber *> *)layerOutputs learningRate:(double)learningRate
{
    // L         : learning rate
    // Error(k)  : Delta(k)
    // Output(j) : Output of Net of Last Layer
    // b(k)      : bias(k)
    // w(jk)     : weight(jk), e.g. net(2) to net(3) = w(23)
    
    // Looping all output nets.
    [_nets enumerateObjectsUsingBlock:^(KRMLPHiddenNet * _Nonnull hiddenNet, NSUInteger idx, BOOL * _Nonnull stop) {
        // Updating biases.
        double newBias                   = [hiddenNet newBiasWithLearningRate:learningRate];
        hiddenNet.bias                   = newBias;
        
        // Updating weights.
        NSArray <NSNumber *> *newWeights = [hiddenNet newWeightsWithLayerOutputs:layerOutputs learningRate:learningRate];
        [hiddenNet renewWeights:newWeights];
    }];
}

#pragma mark - Setters
- (void)setActiveFunction:(KRMLPNetActivations)activeFunction
{
    _activeFunction = activeFunction;
    if( _nets )
    {
        [_nets enumerateObjectsUsingBlock:^(KRMLPHiddenNet * _Nonnull obj, NSUInteger idx, BOOL * _Nonnull stop) {
            obj.netActivation = activeFunction;
        }];
    }
}

#pragma mark - NSCopying
- (instancetype)copyWithZone:(NSZone *)zone
{
    KRMLPHiddenLayer *layer = [[KRMLPHiddenLayer alloc] init];
    layer.index             = [_index copy];
    layer.nets              = [[NSMutableArray alloc] initWithArray:_nets copyItems:YES];   
    layer.outputs           = [[NSMutableArray alloc] initWithArray:_outputs copyItems:YES];
    layer.activeFunction    = _activeFunction;
    return layer;
}

#pragma mark - NSCoding
- (void)encodeWithCoder:(NSCoder *)aCoder
{
    [super encodeWithCoder:aCoder];
    [self encodeObject:_index forKey:@"index"];
    [self encodeObject:_nets forKey:@"nets"];
    [self encodeObject:_outputs forKey:@"outputs"];
    [self encodeObject:@(_activeFunction) forKey:@"activeFunction"];
}

- (instancetype)initWithCoder:(NSCoder *)aDecoder
{
    self = [super initWithCoder:aDecoder];
    if(self)
    {
        _index          = [self decodeForKey:@"index"];
        _nets           = [self decodeForKey:@"nets"];
        _outputs        = [self decodeForKey:@"outputs"];
        _activeFunction = [[self decodeForKey:@"activeFunction"] integerValue];
        _mathLib        = [[KRMathLib alloc] init];
    }
    return self;
}

@end
