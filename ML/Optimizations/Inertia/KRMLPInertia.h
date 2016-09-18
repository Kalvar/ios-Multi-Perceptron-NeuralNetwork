//
//  KRMLPInertia.h
//  KRMLP
//
//  Created by Kalvar Lin on 2016/9/18.
//  Copyright © 2016年 Kalvar Lin. All rights reserved.
//

@import Foundation;

typedef NS_ENUM(NSInteger, KRMLPInertialMethods)
{
    KRMLPInertialFixedRate = 0, // Fixed inertial rate, 固定慣性項
    KRMLPInertialQuickProp,
    KRMLPInertialRProp
};

@class KRMLPNet;

@interface KRMLPInertia : NSObject

@property (nonatomic, assign) KRMLPInertialMethods inertialMethod;
@property (nonatomic, assign) double inertialRate; // (Kecman, 2001) suggested this inertial rate between 0.5 ~ 0.7.

+ (instancetype)sharedInertia;
- (instancetype)initWithMethod:(KRMLPInertialMethods)method;
- (instancetype)init;

- (double)deltaWeightAtIndex:(NSInteger)weightIndex net:(KRMLPNet *)net mappedOutput:(double)mappedOutput learningRate:(double)learningRate;

@end
