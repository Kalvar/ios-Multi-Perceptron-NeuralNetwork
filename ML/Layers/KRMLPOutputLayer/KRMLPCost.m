//
//  KRMLPOutput.m
//  KRMLP
//
//  Created by Kalvar Lin on 2016/7/10.
//  Copyright © 2016年 Kalvar Lin. All rights reserved.
//

#import "KRMLPCost.h"

@implementation KRMLPCost (Checks)

- (BOOL)canCalculate
{
    return ( self.patternsCount != 0.0f && self.outputsCount != 0.0f );
}

@end

@implementation KRMLPCost

- (instancetype)init
{
    self = [super init];
    if( self )
    {
        _costValue     = 0.0f;
        _patternsCount = 0.0f;
        _outputsCount  = 0.0f;
    }
    return self;
}

#pragma mark - Getters
// e.g. Cost Value = SUM( ( d(j) - y(j) ) ^ 2 ) / ( [patterns count] * [outputNets count] ) * 0.5f
// MSE  (均方誤差)  = Cost Value / (訓練範例數目 x 數出層神經元數目) / 2
// RMSE (均方根誤差) = sqrt(Cost Value / (訓練範例數目 x 數出層神經元數目))
- (double)mse
{
    return [self canCalculate] ? _costValue / (_patternsCount * _outputsCount) * 0.5f : NSNotFound;
}

- (double)rmse
{
    return [self canCalculate] ? sqrt(_costValue / (_patternsCount * _outputsCount)) : NSNotFound;
}

@end
