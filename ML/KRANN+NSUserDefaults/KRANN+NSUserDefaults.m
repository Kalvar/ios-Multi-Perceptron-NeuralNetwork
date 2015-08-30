//
//  KRANN+NSUserDefaults.m
//  ANN V2.1.4
//
//  Created by Kalvar on 2014/5/22.
//  Copyright (c) 2014 - 2015年 Kuo-Ming Lin (Kalvar Lin, ilovekalvar@gmail.com). All rights reserved.
//

#import "KRANN+NSUserDefaults.h"
#import "KRANNTrainedNetwork.h"

@implementation NSUserDefaults (ExtendUsages)

#pragma --mark Gets NSDefault Values
/*
 * @ 取出萬用型態
 */
+(id)defaultValueForKey:(NSString *)_key
{
    return [[NSUserDefaults standardUserDefaults] objectForKey:_key];
}

/*
 * @ 取出 String
 */
+(NSString *)stringValueForKey:(NSString *)_key
{
    return [NSString stringWithFormat:@"%@", [self defaultValueForKey:_key]];
}

/*
 * @ 取出 BOOL
 */
+(BOOL)boolValueForKey:(NSString *)_key
{
    return [[NSUserDefaults standardUserDefaults] boolForKey:_key];
}

/*
 * @ 取出 Dictionary
 */
+(NSDictionary *)dictionaryValueForKey:(NSString *)_key
{
    return [[NSUserDefaults standardUserDefaults] dictionaryForKey:_key];
}

/*
 * @ 取出 ANN Network
 */
+(KRANNTrainedNetwork *)trainedNetworkValueForKey:(NSString *)_key
{
    NSData *_objectData = [self defaultValueForKey:_key];
    if( !_objectData )
    {
        return nil;
    }
    return (KRANNTrainedNetwork *)[NSKeyedUnarchiver unarchiveObjectWithData:_objectData];
}

#pragma --mark Saves NSDefault Values
/*
 * @ 儲存萬用型態
 */
+(void)saveDefaultValue:(id)_value forKey:(NSString *)_forKey
{
    [[NSUserDefaults standardUserDefaults] setObject:_value forKey:_forKey];
    [[NSUserDefaults standardUserDefaults] synchronize];
}

/*
 * @ 儲存 String
 */
+(void)saveStringValue:(NSString *)_value forKey:(NSString *)_forKey
{
    [self saveDefaultValue:_value forKey:_forKey];
}

/*
 * @ 儲存 BOOL
 */
+(void)saveBoolValue:(BOOL)_value forKey:(NSString *)_forKey
{
    [self saveDefaultValue:[NSNumber numberWithBool:_value] forKey:_forKey];
}

/*
 * @ 儲存訓練過後的 ANN Netrowk
 */
+(void)saveTrainedNetwork:(KRANNTrainedNetwork *)_value forKey:(NSString *)_forKey
{
    if( _value )
    {
        [self saveDefaultValue:[NSKeyedArchiver archivedDataWithRootObject:_value] forKey:_forKey];
    }
}

#pragma --mark Removes NSDefault Values
+(void)removeValueForKey:(NSString *)_key
{
    [[NSUserDefaults standardUserDefaults] removeObjectForKey:_key];
    [[NSUserDefaults standardUserDefaults] synchronize];
}

@end