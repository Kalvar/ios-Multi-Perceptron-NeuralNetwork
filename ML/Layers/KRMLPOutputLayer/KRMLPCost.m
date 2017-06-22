//
//  KRMLPOutput.m
//  KRMLP
//
//  Created by Kalvar Lin on 2016/7/10.
//  Copyright © 2016年 Kalvar Lin. All rights reserved.
//

#import "KRMLPCost.h"

@interface KRMLPCost ()

@property (nonatomic, readonly) double costValue;                       // The (targetValue - outputValue)^2.
@property (nonatomic) NSMutableArray <NSArray <NSNumber *> *> *outputs; // The outputs of patterns in this iteration.
@property (nonatomic) NSMutableArray <NSArray <NSNumber *> *> *targets; // The targets of patterns in this iteration.

@end

@implementation KRMLPCost (Checks)

- (BOOL)canCalculate
{
    return (self.patternsCount != 0 && self.outputsCount != 0);
}

@end

@implementation KRMLPCost

- (instancetype)init
{
    self = [super init];
    if( self )
    {
        _outputs = [[NSMutableArray alloc] init];
        _targets = [[NSMutableArray alloc] init];
    }
    return self;
}

- (void)addOutputs:(NSArray<NSNumber *> *)outputs targets:(NSArray<NSNumber *> *)targets
{
    [_outputs addObject:[outputs copy]];
    [_targets addObject:[targets copy]];
}

- (void)removeRecords
{
    [_outputs removeAllObjects];
    [_targets removeAllObjects];
}

#pragma mark - Getters
- (double)costValue
{
    __block NSArray <NSArray <NSNumber *> *> *targets = _targets;
    __block double costValue = 0.0f;
    [_outputs enumerateObjectsUsingBlock:^(NSArray<NSNumber *> * _Nonnull patternOutputs, NSUInteger idx, BOOL * _Nonnull stop) {
        __block NSArray <NSNumber *> *patternTargets = [targets objectAtIndex:idx];
        [patternOutputs enumerateObjectsUsingBlock:^(NSNumber * _Nonnull netOutput, NSUInteger idx, BOOL * _Nonnull stop) {
            double outputValue = [netOutput doubleValue];
            double targetValue = [[patternTargets objectAtIndex:idx] doubleValue];
            double outputError = targetValue - outputValue;
            // 計算 MSE, RMSE 共同使用的總和誤差值 : SUM(error^2)
            costValue += (outputError * outputError);
        }];
    }];
    return costValue;
}

- (NSInteger)patternsCount
{
    return [_outputs count];
}

- (NSInteger)outputsCount
{
    return [[_outputs firstObject] count];
}

// e.g. Cost Value = SUM( ( d(j) - y(j) ) ^ 2 ) / ( [patterns count] * [outputNets count] ) * 0.5f
// MSE  (均方誤差)  = Cost Value / (訓練範例數目 x 數出層神經元數目) / 2
// RMSE (均方根誤差) = sqrt(Cost Value / (訓練範例數目 x 數出層神經元數目))
- (double)mse
{
    return [self canCalculate] ? self.costValue / (self.patternsCount * self.outputsCount) * 0.5f : NSNotFound;
}

- (double)rmse
{
    return [self canCalculate] ? sqrt(self.costValue / (self.patternsCount * self.outputsCount)) : NSNotFound;
}

- (double)crossEntropy
{
    __block NSArray <NSArray <NSNumber *> *> *targets = _targets;
    __block double iterationEntropy = 0.0f;
    __block NSInteger outputCount   = self.outputsCount;
    [_outputs enumerateObjectsUsingBlock:^(NSArray<NSNumber *> * _Nonnull patternOutputs, NSUInteger idx, BOOL * _Nonnull stop) {
        __block NSArray <NSNumber *> *patternTargets = [targets objectAtIndex:idx];
        __block double patternEntropy = 0.0f;
        [patternOutputs enumerateObjectsUsingBlock:^(NSNumber * _Nonnull netOutput, NSUInteger idx, BOOL * _Nonnull stop) {
            double outputValue = [netOutput doubleValue];
            double targetValue = [[patternTargets objectAtIndex:idx] doubleValue];
            double entropy     = (targetValue * log(outputValue)) + ((1.0f - targetValue) * log(1.0f - outputValue));
            patternEntropy    += entropy;
        }];
        iterationEntropy += -(patternEntropy / outputCount);
    }];
    return (iterationEntropy / self.patternsCount);
}

@end
