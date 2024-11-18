import json
import re
from typing import Any, Dict, List, Union
from pyspark.sql import SparkSession, DataFrame, functions as F
from pyspark.sql.types import StructType, ArrayType, StringType
from functools import lru_cache

@lru_cache(maxsize=1000)
def clean_string(s: str) -> str:
    """Clean string values by removing extra whitespace. Cached for performance."""
    return re.sub(r"\s+", " ", s).strip()

def clean_structure(item: Any) -> Any:
    """
    Recursively clean dictionaries, lists, and nested structures.
    Removes nulls, empty values, and normalizes strings.
    
    Args:
        item: The item to clean (can be dict, list, str, or other types)
        
    Returns:
        Cleaned version of the input item
    """
    if item is None:
        return None
        
    if isinstance(item, dict):
        cleaned_dict = {}
        for k, v in item.items():
            cleaned_key = clean_string(k) if isinstance(k, str) else k
            if not cleaned_key:
                continue
                
            cleaned_value = clean_structure(v)
            if cleaned_value not in (None, "", [], {}):
                cleaned_dict[cleaned_key] = cleaned_value
        return cleaned_dict or None
        
    if isinstance(item, list):
        cleaned_list = [
            cleaned_item for item_value in item
            if (cleaned_item := clean_structure(item_value)) not in (None, "", [], {})
        ]
        return cleaned_list or None
        
    if isinstance(item, str):
        cleaned_str = clean_string(item)
        return cleaned_str if cleaned_str else None
        
    return item

def create_null_safe_struct(struct_col: str, struct_fields: List) -> F.Column:
    """
    Create a null-safe struct with nested field handling.
    
    Args:
        struct_col: Name of the struct column
        struct_fields: List of fields in the struct
        
    Returns:
        A Spark Column expression for the null-safe struct
    """
    processed_fields = []
    for field in struct_fields:
        field_path = f"{struct_col}.{field.name}"
        
        if isinstance(field.dataType, StructType):
            # Handle nested struct
            nested_expr = create_null_safe_struct(field_path, field.dataType.fields)
            processed_fields.append(
                F.when(nested_expr.isNotNull(), nested_expr).alias(field.name)
            )
        elif isinstance(field.dataType, ArrayType):
            if isinstance(field.dataType.elementType, StructType):
                # Handle array of structs
                nested_expr = F.expr(f"""
                    transform(
                        filter({field_path}, x -> x is not null),
                        x -> struct({
                            ','.join(f"x.{nested_field.name}" 
                            for nested_field in field.dataType.elementType.fields)
                        })
                    )
                """)
            else:
                # Handle simple arrays
                nested_expr = F.expr(f"filter({field_path}, x -> x is not null)")
            
            processed_fields.append(
                F.when(F.col(field_path).isNotNull(), nested_expr).alias(field.name)
            )
        else:
            # Handle primitive fields
            processed_fields.append(
                F.when(F.col(field_path).isNotNull(), F.col(field_path)).alias(field.name)
            )
    
    return F.struct(*processed_fields)

def recursively_remove_nulls(df: DataFrame) -> DataFrame:
    """
    Recursively remove null values from all columns in the DataFrame,
    including nested structures.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with nulls removed from all levels
    """
    for field in df.schema.fields:
        if isinstance(field.dataType, StructType):
            # Process struct columns
            df = df.withColumn(
                field.name,
                create_null_safe_struct(field.name, field.dataType.fields)
            )
        elif isinstance(field.dataType, ArrayType):
            if isinstance(field.dataType.elementType, StructType):
                # Process array of structs
                df = df.withColumn(
                    field.name,
                    F.expr(f"""
                        transform(
                            filter({field.name}, x -> x is not null),
                            x -> struct({
                                ','.join(f"x.{nested_field.name}" 
                                for nested_field in field.dataType.elementType.fields)
                            })
                        )
                    """)
                )
            else:
                # Process simple arrays
                df = df.withColumn(
                    field.name,
                    F.expr(f"filter({field.name}, x -> x is not null)")
                )
        else:
            # Process primitive columns
            df = df.withColumn(
                field.name,
                F.when(F.col(field.name).isNotNull(), F.col(field.name))
            )

    return df

def clean_nested_structures(df: DataFrame, target_column: str) -> DataFrame:
    """
    Clean nested structures in a DataFrame column by removing nulls and empty values.
    
    Args:
        df: Input DataFrame
        target_column: Column containing nested structures to clean
        
    Returns:
        DataFrame with cleaned nested structures
    """
    try:
        # Convert target column to JSON RDD
        json_rdd = df.select(target_column).toJSON().map(lambda x: json.loads(x))
        
        # Clean the structures
        cleaned_rdd = json_rdd.map(clean_structure)
        
        # Convert back to DataFrame
        cleaned_json_rdd = cleaned_rdd.map(lambda x: json.dumps(x) if x is not None else None)
        cleaned_df = df.sparkSession.read.json(cleaned_json_rdd)
        
        # Apply recursive null removal
        cleaned_df = recursively_remove_nulls(cleaned_df)
        
        # Replace the cleaned column in original DataFrame
        return df.withColumn(
            target_column,
            F.struct(*[
                F.col(f"{target_column}.{field.name}")
                for field in df.schema[target_column].dataType.fields
            ])
        )
        
    except Exception as e:
        raise ValueError(f"Error cleaning DataFrame: {str(e)}")

def check_nulls(df: DataFrame, column_path: str) -> int:
    """
    Check the number of nulls in a specific column path.
    
    Args:
        df: DataFrame to check
        column_path: Path to the column (e.g., 'field1.nested1.deep1')
        
    Returns:
        Number of null values found
    """
    return df.select(F.col(column_path)).where(F.col(column_path).isNull()).count()

# Example usage
def main():
    spark = SparkSession.builder.appName("NestedStructureCleaner").getOrCreate()
    
    # Create sample data with nested nulls
    sample_data = [
        {
            "experianResponse": {
                "field1": "value1",
                "field2": None,
                "field3": {
                    "nested1": "data",
                    "nested2": None,
                    "nested3": {
                        "deep1": "value",
                        "deep2": None
                    }
                },
                "arrayField": [
                    {"item1": "value", "item2": None},
                    None,
                    {"item1": None, "item2": "value"}
                ]
            }
        }
    ]
    
    # Create DataFrame and clean it
    df = spark.createDataFrame(sample_data)
    cleaned_df = clean_nested_structures(df, "experianResponse")
    
    # Show results
    cleaned_df.printSchema()
    cleaned_df.show(truncate=False)
    
    # Check for remaining nulls
    null_count = check_nulls(cleaned_df, "experianResponse.field3.nested2")
    print(f"Number of nulls in nested2: {null_count}")

if __name__ == "__main__":
    main()
