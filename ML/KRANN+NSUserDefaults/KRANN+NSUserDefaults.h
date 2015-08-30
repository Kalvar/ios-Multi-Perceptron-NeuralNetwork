//
//  KRANN+NSUserDefaults.h
//  ANN V2.1.4
//
//  Created by Kalvar on 2014/5/22.
//  Copyright (c) 2014 - 2015年 Kuo-Ming Lin (Kalvar Lin, ilovekalvar@gmail.com). All rights reserved.
//

#import <Foundation/Foundation.h>

@class KRANNTrainedNetwork;

@interface NSUserDefaults (ExtendUsages)

#pragma --mark Gets NSDefault Values
+(id)defaultValueForKey:(NSString *)_key;
+(NSString *)stringValueForKey:(NSString *)_key;
+(BOOL)boolValueForKey:(NSString *)_key;
+(NSDictionary *)dictionaryValueForKey:(NSString *)_key;
+(KRANNTrainedNetwork *)trainedNetworkValueForKey:(NSString *)_key;

#pragma --mark Saves NSDefault Values
+(void)saveDefaultValue:(id)_value forKey:(NSString *)_forKey;
+(void)saveStringValue:(NSString *)_value forKey:(NSString *)_forKey;
+(void)saveBoolValue:(BOOL)_value forKey:(NSString *)_forKey;
+(void)saveTrainedNetwork:(KRANNTrainedNetwork *)_value forKey:(NSString *)_forKey;

#pragma --mark Removes NSDefault Values
+(void)removeValueForKey:(NSString *)_key;

@end
