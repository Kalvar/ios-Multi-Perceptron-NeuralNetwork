//
//  KRMLPOptimization.m
//  KRMLP
//
//  Created by Kalvar Lin on 2016/8/30.
//  Copyright © 2016年 Kalvar Lin. All rights reserved.
//

#import "KRMLPOptimization.h"
#import "KRMLPNet.h"
#import "KRMLPInertia.h"

@implementation KRMLPOptimization

+ (instancetype)shared
{
    static dispatch_once_t pred;
    static KRMLPOptimization *object = nil;
    dispatch_once(&pred, ^{
        object = [[KRMLPOptimization alloc] init];
    });
    return object;
}

- (instancetype)initWithMethod:(KRMLPOptimizationMethods)method
{
    self = [super init];
    if( self )
    {
        _method = method;
    }
    return self;
}

- (instancetype)init
{
    return [self initWithMethod:KRMLPOptimizationDefault];
}

#pragma mark - Methods
// weightIndex:  是哪條權重
// net:          要更新的神女經元
// mappedOutput: 該 net 對應的上一層 layer net 的輸出值
// learningRate: 學習速率
- (double)deltaWeightAtIndex:(NSInteger)weightIndex net:(KRMLPNet *)net mappedOutput:(double)mappedOutput learningRate:(double)learningRate
{
    double deltaValue = 0.0f;
    switch (_method)
    {
        case KRMLPOptimizationQuickProp:
        case KRMLPOptimizationFixedInertia:
        {
            KRMLPInertia *inertia = [KRMLPInertia sharedInertia];
            deltaValue = [inertia deltaWeightAtIndex:weightIndex net:net mappedOutput:mappedOutput learningRate:learningRate];
        }
            break;
        case KRMLPOptimizationRProp: // Todolist
        case KRMLPOptimizationEDBD:  // Todolist
        default:
            // Follows the original backpropagation method: L * Error(k) * Output(j) + w(jk)
            deltaValue = learningRate * net.deltaValue * mappedOutput;
            break;
    }
    return deltaValue;
}

#pragma mark - Setters
- (void)setMethod:(KRMLPOptimizationMethods)method
{
    _method = method;
    switch (method)
    {
        case KRMLPOptimizationQuickProp:
            [KRMLPInertia sharedInertia].inertialMethod = KRMLPInertialQuickProp;
            break;
        case KRMLPOptimizationRProp: // Todolist
        case KRMLPOptimizationEDBD:  // Todolist
        default:
            [KRMLPInertia sharedInertia].inertialMethod = KRMLPInertialFixedRate;
            break;
    }
}

- (void)setInertialRate:(double)inertialRate
{
    KRMLPInertia *inertia = [KRMLPInertia sharedInertia];
    inertia.inertialRate  = inertialRate;
}

#pragma mark - Getters
- (double)inertialRate
{
    return [KRMLPInertia sharedInertia].inertialRate;
}

@end
