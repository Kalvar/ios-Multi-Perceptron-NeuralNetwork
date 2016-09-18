//
//  KRMathLib.m
//
//  Created by Kalvar Lin on 2015/9/19.
//  Copyright (c) 2015年 Kalvar Lin. All rights reserved.
//

#import "KRMathLib.h"

@implementation KRMathLib

+(instancetype)sharedLib
{
    static dispatch_once_t pred;
    static KRMathLib *_object = nil;
    dispatch_once(&pred, ^{
        _object = [[KRMathLib alloc] init];
    });
    return _object;
}

-(instancetype)init
{
    self = [super init];
    if( self )
    {
        // ...
    }
    return self;
}

@end

@implementation KRMathLib (RandomNumber)

-(NSInteger)randomIntegerMax:(NSInteger)_maxValue min:(NSInteger)_minValue
{
    return ( arc4random() / ( RAND_MAX * 2.0f ) ) * (_maxValue - _minValue) + _minValue;
    // Randoms value between 0 and 50
    //NSInteger x = arc4random() % 50;
    // Randoms number between 20 and 40
    //NSInteger x = (arc4random() % 21) + 20);
}

-(double)randomDoubleMax:(double)_maxValue min:(double)_minValue
{
    if( _maxValue == _minValue )
    {
        if ( _maxValue == 0.0f )
        {
            return 0.0f;
        }
        else
        {
            return _maxValue;
        }
    }
    return ((double)arc4random() / ( RAND_MAX * 2.0f ) ) * (_maxValue - _minValue) + _minValue;
}

@end

@implementation KRMathLib (ArrayOperations)

-(NSArray *)sortArray:(NSArray *)_array byKey:(NSString *)_byKey ascending:(BOOL)_ascending
{
    return [_array sortedArrayUsingDescriptors:@[[NSSortDescriptor sortDescriptorWithKey:_byKey ascending:_ascending]]];
}

// ex : [1, 2]^T * [3, 4]
-(double)sumMatrix:(NSArray *)_parentMatrix anotherMatrix:(NSArray *)_childMatrix
{
    double _sum      = 0.0f;
    NSInteger _index = 0;
    for( NSNumber *_value in _parentMatrix )
    {
        _sum += [_value doubleValue] * [[_childMatrix objectAtIndex:_index] doubleValue];
        ++_index;
    }
    return _sum;
}

-(double)sumArray:(NSArray *)_array
{
    double _sum = 0.0f;
    for( NSNumber *_value in _array )
    {
        _sum += [_value doubleValue];
    }
    return _sum;
}

// ex : 0.5f * [1, 2]
-(NSArray *)multiplyMatrix:(NSArray *)_matrix byNumber:(double)_number
{
    NSMutableArray *_array = [NSMutableArray new];
    for( NSNumber *_value in _matrix )
    {
        double _newValue = _number * [_value doubleValue];
        [_array addObject:[NSNumber numberWithDouble:_newValue]];
    }
    return _array;
}

// ex : [1, 2] + [3, 4]
-(NSArray *)plusMatrix:(NSArray *)_matrix anotherMatrix:(NSArray *)_anotherMatrix
{
    NSMutableArray *_array = [NSMutableArray new];
    NSInteger _index       = 0;
    for( NSNumber *_value in _matrix )
    {
        double _newValue = [_value doubleValue] + [[_anotherMatrix objectAtIndex:_index] doubleValue];
        [_array addObject:[NSNumber numberWithDouble:_newValue]];
        ++_index;
    }
    return _array;
}

// ex : [1, 2] - [3, 4]
-(NSArray *)minusMatrix:(NSArray *)_matrix anotherMatrix:(NSArray *)_anotherMatrix
{
    NSMutableArray *_array = [NSMutableArray new];
    NSInteger _index       = 0;
    for( NSNumber *_value in _matrix )
    {
        double _newValue = [_value doubleValue] - [[_anotherMatrix objectAtIndex:_index] doubleValue];
        [_array addObject:[NSNumber numberWithDouble:_newValue]];
        ++_index;
    }
    return _array;
}

-(NSMutableArray *)multiplyMatrix:(NSArray *)_parentMatrix anotherMatrix:(NSArray *)_childMatrix
{
    NSMutableArray *_combinedMatrix = [NSMutableArray new];
    for( NSArray *_tRows in _parentMatrix )
    {
        NSMutableArray *_sumRows = [NSMutableArray new];
        for( NSArray *_bRows in _childMatrix )
        {
            NSInteger _index = 0;
            double _sum      = 0.0f;
            for( NSNumber *_bValue in _bRows )
            {
                NSNumber *_tValue  = [_tRows objectAtIndex:_index];
                _sum              += [_tValue doubleValue] * [_bValue doubleValue];
                //NSLog(@"%f x %f", [_tValue doubleValue], [_bValue doubleValue]);
                ++_index;
            }
            //NSLog(@"_sum %f \n\n", _sum);
            [_sumRows addObject:[NSNumber numberWithDouble:_sum]];
        }
        [_combinedMatrix addObject:_sumRows];
    }
    return _combinedMatrix;
}

-(NSMutableArray *)transposeMatrix:(NSArray *)_matrix
{
    // 多維陣列要用多個 Array 互包來完成
    if( !_matrix ) return nil;
    NSMutableArray *_transposedMatrix = [[NSMutableArray alloc] initWithCapacity:0];
    NSInteger _xCount = [_matrix count];
    NSInteger _yCount = 0;
    //如果第 1 個值為陣列
    if( [[_matrix objectAtIndex:0] isKindOfClass:[NSArray class]] )
    {
        //即為 N 維陣列
        _xCount = [[_matrix objectAtIndex:0] count];
        _yCount = [_matrix count];
    }
    
    // 1 維陣列
    if( _yCount == 0 )
    {
        for( int x=0; x<_xCount; x++ )
        {
            [_transposedMatrix addObject:[[NSArray alloc] initWithObjects:[_matrix objectAtIndex:x], nil]];
        }
    }
    else
    {
        for( int x=0; x<_xCount; x++ )
        {
            //轉置，所以 x 總長度為 _yCount
            NSMutableArray *_newRows = [[NSMutableArray alloc] initWithCapacity:_yCount];
            for( int y=0; y<_yCount; y++ )
            {
                //NSLog(@"x = %i, y = %i", x, y);
                if( [[_matrix objectAtIndex:y] isKindOfClass:[NSArray class]] )
                {
                    [_newRows addObject:[[_matrix objectAtIndex:y] objectAtIndex:x]];
                }
                else
                {
                    [_newRows addObject:[_matrix objectAtIndex:y]];
                }
            }
            [_transposedMatrix addObject:_newRows];
        }
    }
    return _transposedMatrix;
}

@end

@implementation KRMathLib (SolveEquations)
// 使用最小平方法(LMS)來求方陣解聯立
// Solves that simultaneous equations
-(NSMutableArray <NSNumber *> *)solveEquationsAtMatrix:(NSArray *)_matrix outputs:(NSArray *)_outputs
{
    NSMutableArray *_solvedEquations = nil;
    
    // Formula is ( B^T x B )^-1 x B^T x Yn
    NSMutableArray *_transposedMatrix = [self transposeMatrix:_matrix];
    // Doing ( B^T x B )
    // 因為陣列取值的順序關係，用同一個轉置矩陣互乘運算即可達到同樣效果，但在手解式上就不能這麼做
    // Ex : [20 x 5] x [5 x 20] = [20 x 20]
    NSMutableArray *_squareMatrixes   = [self multiplyMatrix:_transposedMatrix anotherMatrix:_transposedMatrix];
    // Doing (B^T x Yn)
    NSMutableArray *_bxY              = [self multiplyMatrix:_transposedMatrix anotherMatrix:_outputs];
    NSArray *_yN                      = [[self transposeMatrix:_bxY] firstObject];
    
    @autoreleasepool
    {
        long int rows = [_squareMatrixes count];
        long int cols = [[_squareMatrixes firstObject] count];
        double buffer[ rows * cols ]; // x 直欄(rows) y 橫列(clos)
        int i         = 0;
        // Transforms OC array into C array
        for( NSArray *_xRows in _squareMatrixes )
        {
            for( NSNumber *_number in _xRows )
            {
                buffer[i] = [_number doubleValue];
                ++i;
            }
        }
        
        // array, x 欄(行) y 列
        la_object_t matrix = la_matrix_from_double_buffer(buffer, rows, cols, cols, LA_NO_HINT, LA_ATTRIBUTE_ENABLE_LOGGING);
        
        //NSLog(@"rows %li", la_matrix_rows(matrix));
        //NSLog(@"cols %li", la_matrix_cols(matrix));
        
        long int bRows     = [_yN count];
        long int bCols     = 1;
        double buffer2[ bRows * bCols ];
        int j              = 0;
        for( NSNumber *_outputValue in _yN )
        {
            buffer2[j] = [_outputValue doubleValue];
            ++j;
        }
        
        la_object_t vector = la_matrix_from_double_buffer(buffer2, bRows, bCols, bCols, LA_NO_HINT, LA_ATTRIBUTE_ENABLE_LOGGING);
        
        // 解出來的聯立解
        la_object_t solvedMatrix = la_solve(matrix, vector);
        double solvedBuffer[ bRows * bCols ];
        
        // 從Matrix物件 轉回 double[]
        la_status_t status = la_matrix_to_double_buffer(solvedBuffer, 1, solvedMatrix);
        
        if (status == LA_SUCCESS)
        {
            _solvedEquations = [NSMutableArray new];
            for( int i=0; i<bRows * bCols; i++ )
            {
                [_solvedEquations addObject:[NSNumber numberWithDouble:solvedBuffer[i]]];
            }
        }
        
    }
    return _solvedEquations;
}

@end

@implementation KRMathLib (Distance)

-(double)distance:(NSArray *)_x1 x2:(NSArray *)_x2
{
    NSInteger _index = 0;
    double _sum      = 0.0f;
    for( NSNumber *_x in _x1 )
    {
        _sum        += powf([_x doubleValue] - [[_x2 objectAtIndex:_index] doubleValue], 2);
        ++_index;
    }
    return _sum;
}

-(double)euclidean:(NSArray *)_x1 x2:(NSArray *)_x2
{
    double _sum = [self distance:_x1 x2:_x2];
    return (_sum > 0.0f) ? sqrtf(_sum) : _sum;
}

@end