//
//  KRANNTrainedNetwork.m
//  ANN V2.1.4
//
//  Created by Kalvar on 2014/5/22.
//  Copyright (c) 2014 - 2015年 Kuo-Ming Lin (Kalvar Lin, ilovekalvar@gmail.com). All rights reserved.
//

#import "KRANNTrainedNetwork.h"

@interface KRANNTrainedNetwork ()

@property (nonatomic, weak) NSCoder *_coder;

@end

@implementation KRANNTrainedNetwork (fixNSCodings)

-(void)_encodeObject:(id)_object forKey:(NSString *)_key
{
    [self._coder encodeObject:_object forKey:_key];
}

-(void)_encodeDouble:(double)_object forKey:(NSString *)_key
{
    [self._coder encodeDouble:_object forKey:_key];
}

-(void)_encodeFloat:(float)_object forKey:(NSString *)_key
{
    [self._coder encodeFloat:_object forKey:_key];
}

-(void)_encodeInteger:(NSInteger)_object forKey:(NSString *)_key
{
    [self._coder encodeInteger:_object forKey:_key];
}

@end

@implementation KRANNTrainedNetwork

@synthesize inputs             = _inputs;
@synthesize inputWeights       = _inputWeights;
@synthesize hiddenLayers       = _hiddenLayers;
@synthesize allHiddenWeights   = _allHiddenWeights;
@synthesize allHiddenBiases    = _allHiddenBiases;
@synthesize outputBiases       = _outputBiases;
@synthesize outputResults      = _outputResults;
@synthesize outputGoals        = _outputGoals;
@synthesize learningRate       = _learningRate;
@synthesize convergenceError   = _convergenceError;
@synthesize fOfAlpha           = _fOfAlpha;
@synthesize limitIteration    = _limitIteration;
@synthesize trainingIteration = _trainingIteration;

+(instancetype)sharedNetwork
{
    static dispatch_once_t pred;
    static KRANNTrainedNetwork *_object = nil;
    dispatch_once(&pred, ^{
        _object = [[KRANNTrainedNetwork alloc] init];
    });
    return _object;
}

-(instancetype)init
{
    self = [super init];
    if( self )
    {
        
    }
    return self;
}

#pragma --mark NSCoding Auto Lifecycle
/*
 * @ 以下在呼叫 [self init] 時就會被自動建立
 */
-(void)encodeWithCoder:(NSCoder *)aCoder
{
    self._coder = aCoder;
    
    [self _encodeObject:_inputs forKey:@"inputs"];
    [self _encodeObject:_inputWeights forKey:@"inputWeights"];
    [self _encodeObject:_hiddenLayers forKey:@"hiddenLayers"];
    [self _encodeObject:_allHiddenWeights forKey:@"allHiddenWeights"];
    [self _encodeObject:_allHiddenBiases forKey:@"allHiddenBiases"];
    
    [self _encodeObject:_outputBiases forKey:@"outputBiases"];
    [self _encodeObject:_outputResults forKey:@"outputResults"];
    [self _encodeObject:_outputGoals forKey:@"outputGoals"];
    
    [self _encodeFloat:_learningRate forKey:@"learningRate"];
    [self _encodeDouble:_convergenceError forKey:@"convergenceError"];
    [self _encodeFloat:_fOfAlpha forKey:@"fOfAlpha"];
    
    [self _encodeInteger:_limitIteration forKey:@"limitIteration"];
    [self _encodeInteger:_trainingIteration forKey:@"trainingIteration"];
}

-(instancetype)initWithCoder:(NSCoder *)aDecoder
{
    self = [super init];
    if(self)
    {
        self._coder         = aDecoder;
        
        _inputs             = [aDecoder decodeObjectForKey:@"inputs"];
        _inputWeights       = [aDecoder decodeObjectForKey:@"inputWeights"];
        _hiddenLayers       = [aDecoder decodeObjectForKey:@"hiddenLayers"];
        _allHiddenWeights   = [aDecoder decodeObjectForKey:@"allHiddenWeights"];
        _allHiddenBiases    = [aDecoder decodeObjectForKey:@"allHiddenBiases"];
        
        _outputBiases       = [aDecoder decodeObjectForKey:@"outputBiases"];
        _outputResults      = [aDecoder decodeObjectForKey:@"outputResults"];
        _outputGoals        = [aDecoder decodeObjectForKey:@"outputGoals"];
        
        _learningRate       = [aDecoder decodeFloatForKey:@"learningRate"];
        _convergenceError   = [aDecoder decodeDoubleForKey:@"convergenceError"];
        _fOfAlpha           = [aDecoder decodeFloatForKey:@"fOfAlpha"];
        
        _limitIteration    = [aDecoder decodeIntegerForKey:@"limitIteration"];
        _trainingIteration = [aDecoder decodeIntegerForKey:@"trainingIteration"];
        
    }
    return self;
}

@end
