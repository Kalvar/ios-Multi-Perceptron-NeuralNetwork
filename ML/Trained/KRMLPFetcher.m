//
//  KRRBFFetcher.m
//  KRMLP
//
//  Created by Kalvar Lin on 2016/4/14.
//  Copyright © 2016年 Kalvar Lin. All rights reserved.
//

#import "KRMLPFetcher.h"

@implementation KRMLPFetcher

+ (instancetype)sharedFetcher
{
    static dispatch_once_t pred;
    static KRMLPFetcher *_object = nil;
    dispatch_once(&pred, ^{
        _object = [[KRMLPFetcher alloc] init];
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

- (void)save:(KRMLPPassed *)object forKey:(NSString *)key
{
    if( object && key )
    {
        [[NSUserDefaults standardUserDefaults] setObject:[NSKeyedArchiver archivedDataWithRootObject:object] forKey:key];
        [[NSUserDefaults standardUserDefaults] synchronize];
    }
}

- (void)removeForKey:(NSString *)key
{
    if( key )
    {
        [[NSUserDefaults standardUserDefaults] removeObjectForKey:key];
        [[NSUserDefaults standardUserDefaults] synchronize];
    }
}

// Fetching saved object that all recoreded parameters from trained network.
- (KRMLPPassed *)objectForKey:(NSString *)key
{
    if( key )
    {
        NSData *_objectData = [[NSUserDefaults standardUserDefaults] valueForKey:key];
        return _objectData ? [NSKeyedUnarchiver unarchiveObjectWithData:_objectData] : nil;
    }
    return nil;
}

@end
