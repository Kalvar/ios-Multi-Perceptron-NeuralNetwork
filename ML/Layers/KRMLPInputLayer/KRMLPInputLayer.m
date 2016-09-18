//
//  KRMLPInputLayer.m
//  KRMLP
//
//  Created by Kalvar Lin on 2016/6/19.
//  Copyright © 2016年 Kalvar Lin. All rights reserved.
//

#import "KRMLPInputLayer.h"
#import "KRMLPPattern.h"

@implementation KRMLPInputLayer

- (instancetype)init
{
    self = [super init];
    if( self )
    {
        _nets = [NSMutableArray new];
    }
    return self;
}

- (void)addNet:(KRMLPPattern *)net
{
    [_nets addObject:net];
}

#pragma mark - Getters
- (NSInteger)inputNetsCount
{
    return ( [_nets count] > 0 )? [[_nets firstObject].features count] : 0;
}

@end
