//
//  KRMLPInertia.m
//  KRMLP
//
//  Created by Kalvar Lin on 2016/9/18.
//  Copyright © 2016年 Kalvar Lin. All rights reserved.
//

#import "KRMLPInertia.h"
#import "KRMLPNet.h"

@implementation KRMLPInertia

+ (instancetype)sharedInertia
{
    static dispatch_once_t pred;
    static KRMLPInertia *object = nil;
    dispatch_once(&pred, ^{
        object = [[KRMLPInertia alloc] init];
    });
    return object;
}

- (instancetype)initWithMethod:(KRMLPInertialMethods)method
{
    self = [super init];
    if( self )
    {
        _inertialRate   = 0.75f;  // My suggesed.
        _inertialMethod = method;
    }
    return self;
}

- (instancetype)init
{
    return [self initWithMethod:KRMLPInertialFixedRate];
}

- (double)deltaWeightAtIndex:(NSInteger)weightIndex net:(KRMLPNet *)net mappedOutput:(double)mappedOutput learningRate:(double)learningRate
{
    double deltaValue = learningRate * net.deltaValue * mappedOutput;
    if( net.updatedTimes > 0 )
    {
        double lastDeltaWeight = [[net.lastDeltaWeights objectAtIndex:weightIndex] doubleValue];
        switch (_inertialMethod)
        {
            case KRMLPInertialQuickProp:
            {
                // Grandient Rate of QuickProp = -(net.deltaValue * mappedOutput)
                // Fetchs last grandient rate (上次的梯度下降率):
                double lastGradientValue   = [[net.lastGradients objectAtIndex:weightIndex] doubleValue];
                
                // Current grandient value:
                double gradientValue       = [[net.gradients objectAtIndex:weightIndex] doubleValue];
                
                // Dynamic learning rate is QuickProp.
                double diffGradient        = lastGradientValue - gradientValue;
                double dynamicLearningRate = _inertialRate;
                // 前後次下降差值(除數)不為 0
                if( diffGradient != 0.0f )
                {
                    dynamicLearningRate = gradientValue / diffGradient;
                    // Fahlman suggested the QuickProp learning rate to under 1.75.
                    if( dynamicLearningRate > 1.75f )
                    {
                        dynamicLearningRate = 1.75f;
                    }
                }
                else if( gradientValue == 0.0f )
                {
                    dynamicLearningRate = 0.0f;
                }
                deltaValue += dynamicLearningRate * lastDeltaWeight;
            }
                break;
            case KRMLPInertialRProp: // Todolist, the RProp is better than QuickProp, since QuickProp has overfitting problem.
            case KRMLPInertialFixedRate:
            default:
                // delta w(ji) = L * delta value * y + fixed inertial rate * last delta w(ji)
                deltaValue += _inertialRate * lastDeltaWeight;
                break;
        }
    }
    else
    {
        // 是初次更新權重的狀態
        // 直接照 Backpropagation 原始未套入優化公式的方式去做即可
        // deltaValue = learningRate * net.deltaValue * mappedOutput;
    }
    return deltaValue;
}

@end
