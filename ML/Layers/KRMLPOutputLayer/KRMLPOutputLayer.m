//
//  KRMLPOutputLayer.m
//  KRMLP
//
//  Created by Kalvar Lin on 2016/6/19.
//  Copyright © 2016年 Kalvar Lin. All rights reserved.
//

#import "KRMLPOutputLayer.h"
#import "KRMathLib.h"

@interface KRMLPOutputLayer ()

@property (nonatomic, strong) KRMathLib *mathLib;

@end

@implementation KRMLPOutputLayer

- (instancetype)initWithNets:(NSArray <KRMLPOutputNet *> *)layerNets
{
    self = [super init];
    if( self )
    {
        _nets    = [NSMutableArray new];
        [self addNetsFromArray:layerNets];
        _outputs = [NSMutableArray new];
        _mathLib = [[KRMathLib alloc] init];
    }
    return self;
}

- (instancetype)init
{
    return [self initWithNets:nil];
}

- (void)addNet:(KRMLPOutputNet *)net
{
     if( net )
     {
         [_nets addObject:net];
     }
}

- (void)addNetsFromArray:(NSArray<KRMLPOutputNet *> *)layerNets
{
    if( layerNets )
    {
        [_nets addObjectsFromArray:layerNets];
    }
}

// 計算輸出層的所有輸出結果
// Caculating the outputs of output-layer.
- (NSArray <NSNumber *> *)directOutputsWithInputs:(NSArray <NSNumber *> *)inputs
{
    __block NSMutableArray *outputs = [NSMutableArray new];
    [_nets enumerateObjectsUsingBlock:^(KRMLPOutputNet * _Nonnull net, NSUInteger idx, BOOL * _Nonnull stop) {
        [outputs addObject:@([net outputWithInputs:inputs])];
    }];
    return outputs;
}

- (NSArray <NSNumber *> *)layerOutputsWithInputs:(NSArray <NSNumber *> *)inputs
{
    [_outputs removeAllObjects];
    [_outputs addObjectsFromArray:[self directOutputsWithInputs:inputs]];
    return _outputs;
}

// 計算修正 Weights 與 Biases 用的 Delta Values 和當前 Pattern 的 Cost Value (網路輸出誤差值) for Cost Function.
// Returns the cost function value of this pattern.
- (double)calculateCostAndDeltaWithTargets:(NSArray <NSNumber *> *)targets
{
    __weak typeof(self) weakSelf = self;
    __block double costValue     = 0.0f;
    [_nets enumerateObjectsUsingBlock:^(KRMLPOutputNet * _Nonnull outputNet, NSUInteger idx, BOOL * _Nonnull stop) {
        __strong typeof(weakSelf) strongSelf = weakSelf;
        double targetValue    = [[targets objectAtIndex:idx] doubleValue];
        double outputValue    = [[strongSelf.outputs objectAtIndex:idx] doubleValue];
        double outputError    = targetValue - outputValue; // d(j) - y(j) = d(j) - f(net(j))
        // 計算 MSE, RMSE 共同使用的總和誤差值 : SUM(error^2)
        costValue            += ( outputError * outputError );
        // a is Partial Derivative simple.
        // Delta Value 是由 a(E)/a(wij) = ( d(j) - f(net(j)) ) * -f'(net(j)) = ( d(j) - y(j) ) * -f'( y(j) )
        // Output Net(j) 的 Delta 輸出誤差由於後續修正權重公式的關係, 會將負號消去, 故這裡先直接將負號消去: ( d(j) - y(j) ) * f'( y(j) )
        outputNet.deltaValue  = ( outputError * outputNet.outputPartialDerivative );
    }];
    return costValue;
}

// Updating the weights and biases with last hidden layer outputs.
- (void)updateWithHiddenLayerOutputs:(NSArray <NSNumber *> *)layerOutputs learningRate:(double)learningRate
{
    // L         : learning rate
    // Error(k)  : Delta(k)
    // Output(j) : Output of Net of Last Layer
    // b(k)      : bias(k)
    // w(jk)     : weight(jk), e.g. net(4) to net(6) = w(46)
    
    // Looping all output nets.
    [_nets enumerateObjectsUsingBlock:^(KRMLPOutputNet * _Nonnull outputNet, NSUInteger idx, BOOL * _Nonnull stop) {
        // Updating biases.
        double newBias                   = [outputNet newBiasWithLearningRate:learningRate];
        outputNet.bias                   = newBias;
        
        // Updating weights.
        NSArray <NSNumber *> *newWeights = [outputNet newWeightsWithLayerOutputs:layerOutputs learningRate:learningRate];
        [outputNet renewWeights:newWeights];
    }];
}

#pragma mark - Setters
- (void)setActiveFunction:(KRMLPNetActivations)activeFunction
{
    _activeFunction = activeFunction;
    if( _nets )
    {
        [_nets enumerateObjectsUsingBlock:^(KRMLPOutputNet * _Nonnull obj, NSUInteger idx, BOOL * _Nonnull stop) {
            obj.netActivation = activeFunction;
        }];
    }
}

#pragma mark - NSCopying
- (instancetype)copyWithZone:(NSZone *)zone
{
    KRMLPOutputLayer *layer = [[KRMLPOutputLayer alloc] init];
    layer.nets              = [[NSMutableArray alloc] initWithArray:_nets copyItems:YES];    // Deeply copying
    layer.outputs           = [[NSMutableArray alloc] initWithArray:_outputs copyItems:YES];
    layer.activeFunction    = _activeFunction;
    return layer;
}

#pragma mark - NSCoding
- (void)encodeWithCoder:(NSCoder *)aCoder
{
    [super encodeWithCoder:aCoder];
    [self encodeObject:_nets forKey:@"nets"];
    [self encodeObject:_outputs forKey:@"outputs"];
    [self encodeObject:@(_activeFunction) forKey:@"activeFunction"];
}

- (instancetype)initWithCoder:(NSCoder *)aDecoder
{
    self = [super initWithCoder:aDecoder];
    if(self)
    {
        _nets           = [self decodeForKey:@"nets"];
        _outputs        = [self decodeForKey:@"outputs"];
        _activeFunction = [[self decodeForKey:@"activeFunction"] integerValue];
        _mathLib        = [[KRMathLib alloc] init];
    }
    return self;
}

@end
