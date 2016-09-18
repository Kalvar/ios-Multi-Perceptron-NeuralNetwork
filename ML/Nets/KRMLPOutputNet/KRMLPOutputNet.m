//
//  KRMLPOutputNet.m
//  KRMLP
//
//  Created by Kalvar Lin on 2016/5/2.
//  Copyright © 2016年 Kalvar Lin. All rights reserved.
//

#import "KRMLPOutputNet.h"

@implementation KRMLPOutputNet

- (instancetype)init
{
    self = [super init];
    if( self )
    {
        _targetValue = 0.0f;
    }
    return self;
}

@end
