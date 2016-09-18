//
//  KRMLPActiviation.h
//  MLP
//
//  Created by Kalvar Lin on 2016/4/27.
//  Copyright © 2016年 Kalvar Lin. All rights reserved.
//
#import <Foundation/Foundation.h>

@interface KRMLPActivation : NSObject

+ (instancetype)sharedActiviation;
- (instancetype)init;

- (double)euclidean:(NSArray *)_x1 x2:(NSArray *)_x2;

- (double)rbf:(double)_x sigma:(double)_sigma;
- (double)sigmoid:(double)_x slope:(double)_slope;
- (double)tanh:(double)_x slope:(double)_slope;
- (double)sgn:(double)_x;
- (double)cubic:(double)_x;

- (double)dashSigmoid:(double)_x slope:(double)_slope;
- (double)dashTanh:(double)_x slope:(double)_slope;
- (double)dashRBF:(double)_x sigma:(double)_sigma;
- (double)dashSgn:(double)_x;
- (double)dashCubic:(double)_x;

@end
