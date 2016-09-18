//
//  KRMLPOptimization.h
//  KRMLP
//
//  Created by Kalvar Lin on 2016/8/30.
//  Copyright © 2016年 Kalvar Lin. All rights reserved.
//

@import Foundation;

@class KRMLPNet;

typedef NS_ENUM(NSInteger, KRMLPOptimizationMethods)
{
    KRMLPOptimizationDefault,           // Only follows the original learning rule.
    KRMLPOptimizationQuickProp,         // The dynamic inertia.
    KRMLPOptimizationFixedInertia,      // The fixed inertia.
    KRMLPOptimizationRProp,             // Todolist the future feature.
    KRMLPOptimizationEDBD               // Todolist the future feature.
};

@interface KRMLPOptimization : NSObject

@property (nonatomic, assign) KRMLPOptimizationMethods method;
@property (nonatomic, assign) double inertialRate; // 慣性項速率

+ (instancetype)shared;
- (instancetype)initWithMethod:(KRMLPOptimizationMethods)method;
- (instancetype)init;

- (double)deltaWeightAtIndex:(NSInteger)weightIndex net:(KRMLPNet *)net mappedOutput:(double)mappedOutput learningRate:(double)learningRate;

@end
