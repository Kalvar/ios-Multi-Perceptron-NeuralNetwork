//
//  KRMLPLayer.m
//  KRMLP
//
//  Created by Kalvar Lin on 2016/8/22.
//  Copyright © 2016年 Kalvar Lin. All rights reserved.
//

#import "KRMLPLayer.h"

@interface KRMLPLayer ()

@property (nonatomic, weak) NSCoder *coder;

@end

@implementation KRMLPLayer (NSCoding)

- (void)encodeObject:(id)object forKey:(NSString *)key
{
    if( nil != object )
    {
        [self.coder encodeObject:object forKey:key];
    }
}

- (id)decodeForKey:(NSString *)key
{
    return [self.coder decodeObjectForKey:key];
}

@end

@implementation KRMLPLayer

- (instancetype)init
{
    self = [super init];
    if( self )
    {
        
    }
    return self;
}

#pragma mark - NSCopying
- (instancetype)copyWithZone:(NSZone *)zone
{
    KRMLPLayer *layer = [[KRMLPLayer alloc] init];
    return layer;
}

#pragma mark - NSCoding
- (void)encodeWithCoder:(NSCoder *)aCoder
{
    self.coder = aCoder;
}

- (instancetype)initWithCoder:(NSCoder *)aDecoder
{
    self = [super init];
    if(self)
    {
        self.coder = aDecoder;
    }
    return self;
}

@end
