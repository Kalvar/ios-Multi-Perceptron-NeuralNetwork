//
//  KRMathLib.h
//
//  Created by Kalvar Lin on 2015/9/19.
//  Copyright (c) 2015年 Kalvar Lin. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <Accelerate/Accelerate.h>

@interface KRMathLib : NSObject

+(instancetype)sharedLib;
-(instancetype)init;

@end

@interface KRMathLib (RandomNumber)

-(NSInteger)randomIntegerMax:(NSInteger)_maxValue min:(NSInteger)_minValue;
-(double)randomDoubleMax:(double)_maxValue min:(double)_minValue;

@end

@interface KRMathLib (ArrayOperations)

-(NSArray *)sortArray:(NSArray *)_array byKey:(NSString *)_byKey ascending:(BOOL)_ascending;

-(double)sumMatrix:(NSArray *)_parentMatrix anotherMatrix:(NSArray *)_childMatrix;
-(double)sumArray:(NSArray *)_array;

-(NSArray *)multiplyMatrix:(NSArray *)_matrix byNumber:(double)_number;
-(NSArray *)plusMatrix:(NSArray *)_matrix anotherMatrix:(NSArray *)_anotherMatrix;
-(NSArray *)minusMatrix:(NSArray *)_matrix anotherMatrix:(NSArray *)_anotherMatrix;

-(NSMutableArray *)multiplyMatrix:(NSArray *)_parentMatrix anotherMatrix:(NSArray *)_childMatrix;
-(NSMutableArray *)transposeMatrix:(NSArray *)_matrix;

@end

@interface KRMathLib (SolveEquations)

-(NSMutableArray <NSNumber *> *)solveEquationsAtMatrix:(NSArray *)_matrix outputs:(NSArray *)_outputs;

@end

@interface KRMathLib (Distance)

-(double)distance:(NSArray *)_x1 x2:(NSArray *)_x2; // It's euclidean without sqrt(), the formula is ||x1 - x2||^2
-(double)euclidean:(NSArray *)_x1 x2:(NSArray *)_x2;

@end