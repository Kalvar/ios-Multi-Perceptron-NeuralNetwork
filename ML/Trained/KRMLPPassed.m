//
//  KRRBFPassed.m
//  KRMLP
//
//  Created by Kalvar Lin on 2016/4/12.
//  Copyright © 2016年 Kalvar Lin. All rights reserved.
//

#import "KRMLPPassed.h"
#import "KRMLPHiddenLayer.h"
#import "KRMLPOutputLayer.h"

@interface KRMLPPassed ()

@property (nonatomic, weak) NSCoder *coder;

@end

@implementation KRMLPPassed (Coding)

- (void)_encodeObject:(id)_object forKey:(NSString *)_key
{
    if( nil != _object )
    {
        [self.coder encodeObject:_object forKey:_key];
    }
}

- (id)_decodeForKey:(NSString *)_key
{
    return [self.coder decodeObjectForKey:_key];
}

@end

@implementation KRMLPPassed

-(instancetype)init
{
    self = [super init];
    if( self )
    {
        _hiddenLayers      = [NSMutableArray new];
        _outputLayer       = nil;
        _networkActivation = KRMLPNetActivationDefault;
        _learningRate      = 0.8f;
    }
    return self;
}

#pragma --mark NSCoding
-(void)encodeWithCoder:(NSCoder *)aCoder
{
    self.coder = aCoder;
    [self _encodeObject:_hiddenLayers forKey:@"hiddenLayers"];
    [self _encodeObject:_outputLayer forKey:@"outputLayer"];
    [self _encodeObject:@(_networkActivation) forKey:@"networkActivation"];
    [self _encodeObject:@(_learningRate) forKey:@"learningRate"];
}

-(instancetype)initWithCoder:(NSCoder *)aDecoder
{
    self = [super init];
    if(self)
    {
        self.coder         = aDecoder;
        _hiddenLayers      = [self _decodeForKey:@"hiddenLayers"];
        _outputLayer       = [self _decodeForKey:@"outputLayer"];
        _networkActivation = [[self _decodeForKey:@"networkActivation"] integerValue];
        _learningRate      = [[self _decodeForKey:@"learningRate"] doubleValue];
    }
    return self;
}

@end
