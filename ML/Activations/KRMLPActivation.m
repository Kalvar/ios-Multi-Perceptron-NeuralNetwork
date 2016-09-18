//
//  KRMLPActiviation.m
//  MLP
//
//  Created by Kalvar Lin on 2016/4/27.
//  Copyright © 2016年 Kalvar Lin. All rights reserved.
//

#import "KRMLPActivation.h"

@implementation KRMLPActivation

+ (instancetype)sharedActiviation
{
    static dispatch_once_t pred;
    static KRMLPActivation *_object = nil;
    dispatch_once(&pred, ^{
        _object = [[KRMLPActivation alloc] init];
    });
    return _object;
}

- (instancetype)init
{
    self = [super init];
    if( self )
    {
        
    }
    return self;
}

#pragma mark - Distance
- (double)euclidean:(NSArray *)_x1 x2:(NSArray *)_x2
{
    NSInteger _index = 0;
    double _sum      = 0.0f;
    for( NSNumber *_x in _x1 )
    {
        _sum        += powf([_x doubleValue] - [[_x2 objectAtIndex:_index] doubleValue], 2);
        ++_index;
    }
    return (_index > 0) ? sqrtf(_sum) : _sum;
}

#pragma mark - Activations
- (double)rbf:(double)_x sigma:(double)_sigma
{
    // Formula : exp^( -s / ( 2.0f * sigma * sigma ) )
    return pow(M_E, ((-_x) / ( 2.0f * _sigma * _sigma )));
}

// S 形函數 (Sigmoid), [0.0, 1.0]
- (double)sigmoid:(double)_x slope:(double)_slope
{
    return ( 1.0f / ( 1.0f + pow(M_E, (- (_slope) * _x)) ) );
}

// 雙曲線正切函數 (Hyperbolic Tangent), [-1.0, 1.0]
// the 入 (fofAlpha) standard is 2.0f, but we recommend to use 1.0f and to change the derivative be " (1 - y^2) * 0.5 ".
// the formula is " ( 2.0 / (1.0 + e^(-slope * x)) ) - 1.0 "
- (double)tanh:(double)_x slope:(double)_slope
{
    return ( 2.0f / ( 1.0f + pow(M_E, (- (_slope) * _x)) ) ) - 1.0f;
}

- (double)sgn:(double)_x
{
    return ( _x >= 0.0f ) ? 1.0f : -1.0f;
}

- (double)cubic:(double)_x
{
    return _x * _x * _x;
}

#pragma mark - Partial Differentials (Dash)
- (double)dashSigmoid:(double)_x slope:(double)_slope
{
    // Formula is slope * y * (1 - y)
    return _slope * _x * ( 1.0f - _x );
}

- (double)dashTanh:(double)_x slope:(double)_slope
{
    // Formula : (slope / 2) * (1 - y^2)
    return (_slope / 2.0f) * ( 1.0f - ( _x * _x ) );
}

- (double)dashRBF:(double)_x sigma:(double)_sigma
{
    return - ((2.0 * _x) / (2.0 * _sigma * _sigma)) * pow(M_E, (-_x) / (2.0 * _sigma * _sigma));
}

- (double)dashSgn:(double)_x
{
    return _x;
}

- (double)dashCubic:(double)_x
{
    // Formula : 3(x^2)
    return 3.0f * (_x * _x);
}

@end
