//
//  KRMLPPattern.m
//  KRMLP
//
//  Created by Kalvar Lin on 2016/6/19.
//  Copyright © 2016年 Kalvar Lin. All rights reserved.
//

#import "KRMLPPattern.h"

@implementation KRMLPPattern

- (instancetype)initWithFeatures:(NSArray <NSNumber *> *)f targets:(NSArray <NSNumber *> *)t
{
    self = [super init];
    if( self )
    {
        _features = [NSMutableArray new];
        _targets  = [NSMutableArray new];
        
        [self addFeaturesFromArray:f];
        [self addTargetsFromArray:t];
    }
    return self;
}

- (instancetype)init
{
    return [self initWithFeatures:nil targets:nil];
}

// To copy the feature to control the memory by manually operation is safety.
- (void)addFeature:(NSNumber *)aFeature
{
    if( nil == aFeature )
    {
        return;
    }
    
    [_features addObject:[aFeature copy]];
}

- (void)addFeaturesFromArray:(NSArray *)theFeatures
{
    if( nil == theFeatures )
    {
        return;
    }
    
    for( NSNumber *feature in theFeatures )
    {
        [self addFeature:feature];
    }
}

- (void)addTarget:(NSNumber *)aTarget
{
    if( nil == aTarget )
    {
        return;
    }
    
    [_targets addObject:[aTarget copy]];
}

- (void)addTargetsFromArray:(NSArray *)theTargets
{
    if( nil == theTargets )
    {
        return;
    }
    
    for( NSNumber *target in theTargets )
    {
        [self addTarget:target];
    }
}

@end
