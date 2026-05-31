//! Tool argument validation helpers.

use serde_json::{Map, Number, Value};

use crate::error::{PiError, Result};
use crate::types::{ToolCall, ToolDefinition};

/// Find a tool by name and validate a tool call's JSON arguments.
pub fn validate_tool_call(tools: &[ToolDefinition], tool_call: &ToolCall) -> Result<Value> {
    let tool = tools
        .iter()
        .find(|candidate| candidate.name == tool_call.function.name)
        .ok_or_else(|| PiError::Tool {
            tool: tool_call.function.name.clone(),
            message: format!("Tool \"{}\" not found", tool_call.function.name),
        })?;
    let arguments = serde_json::from_str(&tool_call.function.arguments)?;

    validate_tool_arguments(tool, &arguments)
}

/// Validate tool arguments against the tool's JSON schema parameters.
///
/// The validator mirrors the TypeScript plain-schema behavior used by the
/// existing agent: primitive values are coerced before validation where the
/// coercion is unambiguous.
pub fn validate_tool_arguments(tool: &ToolDefinition, arguments: &Value) -> Result<Value> {
    let coerced = coerce_with_json_schema(arguments.clone(), &tool.parameters);
    let mut errors = Vec::new();
    validate_value(&coerced, &tool.parameters, "root", &mut errors);

    if errors.is_empty() {
        return Ok(coerced);
    }

    Err(PiError::Tool {
        tool: tool.name.clone(),
        message: format!(
            "Validation failed for tool \"{}\":\n{}\n\nReceived arguments:\n{}",
            tool.name,
            errors
                .iter()
                .map(|error| format!("  - {error}"))
                .collect::<Vec<_>>()
                .join("\n"),
            serde_json::to_string_pretty(arguments).unwrap_or_else(|_| arguments.to_string())
        ),
    })
}

fn schema_object(schema: &Value) -> Option<&Map<String, Value>> {
    schema.as_object()
}

fn schema_types(schema: &Value) -> Vec<&str> {
    let Some(object) = schema_object(schema) else {
        return Vec::new();
    };
    match object.get("type") {
        Some(Value::String(schema_type)) => vec![schema_type.as_str()],
        Some(Value::Array(schema_types)) => schema_types
            .iter()
            .filter_map(Value::as_str)
            .collect::<Vec<_>>(),
        _ => Vec::new(),
    }
}

fn matches_json_type(value: &Value, schema_type: &str) -> bool {
    match schema_type {
        "number" => value.is_number(),
        "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
        "boolean" => value.is_boolean(),
        "string" => value.is_string(),
        "null" => value.is_null(),
        "array" => value.is_array(),
        "object" => value.is_object(),
        _ => false,
    }
}

fn number_value(parsed: f64) -> Option<Value> {
    if !parsed.is_finite() {
        return None;
    }
    if parsed.fract() == 0.0 && parsed >= i64::MIN as f64 && parsed <= i64::MAX as f64 {
        return Some(Value::Number(Number::from(parsed as i64)));
    }
    Number::from_f64(parsed).map(Value::Number)
}

fn coerce_primitive_by_type(value: &Value, schema_type: &str) -> Value {
    match schema_type {
        "number" => match value {
            Value::Null => Value::Number(Number::from(0)),
            Value::String(text) if !text.trim().is_empty() => text
                .parse::<f64>()
                .ok()
                .and_then(number_value)
                .unwrap_or_else(|| value.clone()),
            Value::Bool(flag) => Value::Number(Number::from(u8::from(*flag))),
            _ => value.clone(),
        },
        "integer" => match value {
            Value::Null => Value::Number(Number::from(0)),
            Value::String(text) if !text.trim().is_empty() => text
                .parse::<i64>()
                .ok()
                .map(Number::from)
                .map(Value::Number)
                .unwrap_or_else(|| value.clone()),
            Value::Bool(flag) => Value::Number(Number::from(u8::from(*flag))),
            _ => value.clone(),
        },
        "boolean" => match value {
            Value::Null => Value::Bool(false),
            Value::String(text) if text == "true" => Value::Bool(true),
            Value::String(text) if text == "false" => Value::Bool(false),
            Value::Number(number) if number.as_i64() == Some(1) => Value::Bool(true),
            Value::Number(number) if number.as_i64() == Some(0) => Value::Bool(false),
            _ => value.clone(),
        },
        "string" => match value {
            Value::Null => Value::String(String::new()),
            Value::Number(number) => Value::String(number.to_string()),
            Value::Bool(flag) => Value::String(flag.to_string()),
            _ => value.clone(),
        },
        "null" => match value {
            Value::String(text) if text.is_empty() => Value::Null,
            Value::Number(number)
                if number.as_i64() == Some(0)
                    || number.as_u64() == Some(0)
                    || number.as_f64() == Some(0.0) =>
            {
                Value::Null
            }
            Value::Bool(false) => Value::Null,
            _ => value.clone(),
        },
        _ => value.clone(),
    }
}

fn coerce_with_union_schema(value: &Value, schemas: &[Value]) -> Value {
    for schema in schemas {
        let coerced = coerce_with_json_schema(value.clone(), schema);
        let mut errors = Vec::new();
        validate_value(&coerced, schema, "root", &mut errors);
        if errors.is_empty() {
            return coerced;
        }
    }
    value.clone()
}

fn coerce_object_properties(value: &mut Map<String, Value>, schema: &Value) {
    let Some(schema) = schema_object(schema) else {
        return;
    };
    let properties = schema.get("properties").and_then(Value::as_object);
    let defined_keys = properties
        .map(|properties| properties.keys().cloned().collect::<Vec<_>>())
        .unwrap_or_default();

    if let Some(properties) = properties {
        for (key, property_schema) in properties {
            let Some(property_value) = value.get(key).cloned() else {
                continue;
            };
            value.insert(
                key.clone(),
                coerce_with_json_schema(property_value, property_schema),
            );
        }
    }

    let additional_properties = schema
        .get("additionalProperties")
        .filter(|additional| additional.is_object());
    if let Some(additional_properties) = additional_properties {
        for key in value.keys().cloned().collect::<Vec<_>>() {
            if defined_keys.contains(&key) {
                continue;
            }
            let Some(property_value) = value.get(&key).cloned() else {
                continue;
            };
            value.insert(
                key,
                coerce_with_json_schema(property_value, additional_properties),
            );
        }
    }
}

fn coerce_array_items(value: &mut [Value], schema: &Value) {
    let Some(items_schema) = schema_object(schema).and_then(|schema| schema.get("items")) else {
        return;
    };

    match items_schema {
        Value::Array(items) => {
            for (index, item) in value.iter_mut().enumerate() {
                let Some(item_schema) = items.get(index) else {
                    continue;
                };
                *item = coerce_with_json_schema(item.clone(), item_schema);
            }
        }
        Value::Object(_) => {
            for item in value {
                *item = coerce_with_json_schema(item.clone(), items_schema);
            }
        }
        _ => {}
    }
}

fn coerce_with_json_schema(value: Value, schema: &Value) -> Value {
    let mut next_value = value;

    if let Some(all_of) = schema_object(schema).and_then(|schema| schema.get("allOf")) {
        if let Some(nested_schemas) = all_of.as_array() {
            for nested_schema in nested_schemas {
                next_value = coerce_with_json_schema(next_value, nested_schema);
            }
        }
    }

    for key in ["anyOf", "oneOf"] {
        if let Some(nested_schemas) = schema_object(schema)
            .and_then(|schema| schema.get(key))
            .and_then(Value::as_array)
        {
            next_value = coerce_with_union_schema(&next_value, nested_schemas);
        }
    }

    let types = schema_types(schema);
    let matches_union_member = types.len() > 1
        && types
            .iter()
            .any(|schema_type| matches_json_type(&next_value, schema_type));
    if !types.is_empty() && !matches_union_member {
        for schema_type in types {
            let candidate = coerce_primitive_by_type(&next_value, schema_type);
            if candidate != next_value {
                next_value = candidate;
                break;
            }
        }
    }

    let types = schema_types(schema);
    if types.contains(&"object") {
        if let Value::Object(object) = &mut next_value {
            coerce_object_properties(object, schema);
        }
    }

    if types.contains(&"array") {
        if let Value::Array(items) = &mut next_value {
            coerce_array_items(items, schema);
        }
    }

    next_value
}

fn validate_value(value: &Value, schema: &Value, path: &str, errors: &mut Vec<String>) {
    let types = schema_types(schema);
    if !types.is_empty()
        && !types
            .iter()
            .any(|schema_type| matches_json_type(value, schema_type))
    {
        errors.push(format!("{path}: expected {}", types.join(" or ")));
        return;
    }

    let Some(schema) = schema_object(schema) else {
        return;
    };
    // const keyword — exact equality
    if let Some(const_val) = schema.get("const") {
        if value != const_val {
            errors.push(format!("{path}: const mismatch"));
            return;
        }
    }

    // enum keyword — value must be in allowed set
    if let Some(allowed) = schema.get("enum").and_then(Value::as_array) {
        if !allowed.contains(value) {
            errors.push(format!("{path}: value not in enum"));
            return;
        }
    }

    for key in ["anyOf", "oneOf"] {
        if let Some(nested_schemas) = schema.get(key).and_then(Value::as_array) {
            let match_count = nested_schemas
                .iter()
                .filter(|nested_schema| {
                    let mut nested_errors = Vec::new();
                    validate_value(value, nested_schema, path, &mut nested_errors);
                    nested_errors.is_empty()
                })
                .count();

            if key == "oneOf" {
                if match_count == 0 {
                    errors.push(format!("{path}: did not match any oneOf schema"));
                    return;
                } else if match_count > 1 {
                    errors.push(format!("{path}: matched multiple oneOf schemas"));
                    return;
                }
            } else if match_count > 0 {
                return;
            } else {
                errors.push(format!("{path}: did not match any {key} schema"));
                return;
            }
        }
    }

    if let Some(nested_schemas) = schema.get("allOf").and_then(Value::as_array) {
        for nested_schema in nested_schemas {
            validate_value(value, nested_schema, path, errors);
        }
    }

    if let Some(object) = value.as_object() {
        validate_object(object, schema, path, errors);
    }

    if let Some(array) = value.as_array() {
        validate_array(array, schema, path, errors);
    }
    // minLength / maxLength (string)
    if let Some(s) = value.as_str() {
        if let Some(min) = schema.get("minLength").and_then(Value::as_u64) {
            if (s.chars().count() as u64) < min {
                errors.push(format!("{path}: string shorter than minLength {min}"));
            }
        }
        if let Some(max) = schema.get("maxLength").and_then(Value::as_u64) {
            if (s.chars().count() as u64) > max {
                errors.push(format!("{path}: string longer than maxLength {max}"));
            }
        }
    }

    // minimum / maximum (number)
    if let Some(n) = value.as_f64() {
        if let Some(min) = schema.get("minimum").and_then(Value::as_f64) {
            if n < min {
                errors.push(format!("{path}: number below minimum {min}"));
            }
        }
        if let Some(max) = schema.get("maximum").and_then(Value::as_f64) {
            if n > max {
                errors.push(format!("{path}: number above maximum {max}"));
            }
        }
    }
}

fn validate_object(
    value: &Map<String, Value>,
    schema: &Map<String, Value>,
    path: &str,
    errors: &mut Vec<String>,
) {
    if let Some(required) = schema.get("required").and_then(Value::as_array) {
        for required_field in required.iter().filter_map(Value::as_str) {
            if !value.contains_key(required_field) {
                let field_path = if path == "root" {
                    required_field.to_string()
                } else {
                    format!("{path}.{required_field}")
                };
                errors.push(format!("{field_path}: required"));
            }
        }
    }

    if let Some(properties) = schema.get("properties").and_then(Value::as_object) {
        for (key, property_schema) in properties {
            let Some(property_value) = value.get(key) else {
                continue;
            };
            let property_path = if path == "root" {
                key.clone()
            } else {
                format!("{path}.{key}")
            };
            validate_value(property_value, property_schema, &property_path, errors);
        }

        if schema.get("additionalProperties") == Some(&Value::Bool(false)) {
            for key in value.keys() {
                if !properties.contains_key(key) {
                    let property_path = if path == "root" {
                        key.clone()
                    } else {
                        format!("{path}.{key}")
                    };
                    errors.push(format!("{property_path}: unexpected property"));
                }
            }
        }
    }
}

fn validate_array(
    value: &[Value],
    schema: &Map<String, Value>,
    path: &str,
    errors: &mut Vec<String>,
) {
    // minItems / maxItems
    if let Some(min) = schema.get("minItems").and_then(Value::as_u64) {
        if (value.len() as u64) < min {
            errors.push(format!("{path}: array shorter than minItems {min}"));
        }
    }
    if let Some(max) = schema.get("maxItems").and_then(Value::as_u64) {
        if (value.len() as u64) > max {
            errors.push(format!("{path}: array longer than maxItems {max}"));
        }
    }

    let Some(items_schema) = schema.get("items") else {
        return;
    };

    match items_schema {
        Value::Array(items) => {
            for (index, item_schema) in items.iter().enumerate() {
                let Some(item) = value.get(index) else {
                    continue;
                };
                validate_value(item, item_schema, &format!("{path}.{index}"), errors);
            }
        }
        Value::Object(_) => {
            for (index, item) in value.iter().enumerate() {
                validate_value(item, items_schema, &format!("{path}.{index}"), errors);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use crate::FunctionCall;

    fn make_tool(schema: Value) -> ToolDefinition {
        ToolDefinition {
            name: "test_tool".to_string(),
            description: "Test tool".to_string(),
            parameters: schema,
        }
    }

    fn make_tool_call(name: &str, args: Value) -> ToolCall {
        ToolCall {
            id: "call_1".to_string(),
            function: FunctionCall {
                name: name.to_string(),
                arguments: serde_json::to_string(&args).unwrap(),
            },
        }
    }

    // ------------------------------------------------------------------
    // validate_tool_call
    // ------------------------------------------------------------------

    #[test]
    fn validate_tool_call_finds_matching_tool() {
        let tools = vec![make_tool(json!({
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"]
        }))];
        let call = make_tool_call("test_tool", json!({"name": "alice"}));
        let result = validate_tool_call(&tools, &call).unwrap();
        assert_eq!(result, json!({"name": "alice"}));
    }

    #[test]
    fn validate_tool_call_rejects_unknown_tool() {
        let tools = vec![make_tool(json!({"type": "object"}))];
        let call = make_tool_call("nonexistent", json!({}));
        let err = validate_tool_call(&tools, &call).unwrap_err();
        assert!(err.to_string().contains("not found"), "unexpected error: {err}");
    }

    #[test]
    fn validate_tool_call_rejects_malformed_arguments_json() {
        let tools = vec![make_tool(json!({"type": "object"}))];
        let call = ToolCall {
            id: "call_1".to_string(),
            function: FunctionCall {
                name: "test_tool".to_string(),
                arguments: "not valid json".to_string(),
            },
        };
        let err = validate_tool_call(&tools, &call).unwrap_err();
        assert!(err.to_string().contains("serialization error"), "unexpected error: {err}");
    }

    // ------------------------------------------------------------------
    // Required fields
    // ------------------------------------------------------------------

    #[test]
    fn valid_object_with_required_fields_passes() {
        let tool = make_tool(json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"}
            },
            "required": ["name", "age"]
        }));
        let args = json!({"name": "alice", "age": 30});
        let result = validate_tool_arguments(&tool, &args).unwrap();
        assert_eq!(result, args);
    }

    #[test]
    fn missing_required_field_fails() {
        let tool = make_tool(json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"}
            },
            "required": ["name", "age"]
        }));
        let args = json!({"age": 30});
        let err = validate_tool_arguments(&tool, &args).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("required"), "unexpected error: {msg}");
        assert!(msg.contains("name"), "should mention the missing field: {msg}");
    }

    // ------------------------------------------------------------------
    // Type coercion
    // ------------------------------------------------------------------

    #[test]
    fn coercion_string_to_number() {
        let tool = make_tool(json!({
            "type": "object",
            "properties": {"value": {"type": "number"}},
            "required": ["value"]
        }));
        let args = json!({"value": "42"});
        let result = validate_tool_arguments(&tool, &args).unwrap();
        assert_eq!(result["value"], json!(42));
    }

    #[test]
    fn coercion_number_to_string() {
        let tool = make_tool(json!({
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"]
        }));
        let args = json!({"value": 42});
        let result = validate_tool_arguments(&tool, &args).unwrap();
        assert_eq!(result["value"], json!("42"));
    }

    #[test]
    fn coercion_null_to_number() {
        let tool = make_tool(json!({
            "type": "object",
            "properties": {"value": {"type": "number"}},
            "required": ["value"]
        }));
        let args = json!({"value": null});
        let result = validate_tool_arguments(&tool, &args).unwrap();
        assert_eq!(result["value"], json!(0));
    }

    #[test]
    fn coercion_bool_to_number() {
        let tool = make_tool(json!({
            "type": "object",
            "properties": {"value": {"type": "number"}},
            "required": ["value"]
        }));
        assert_eq!(
            validate_tool_arguments(&tool, &json!({"value": true})).unwrap()["value"],
            json!(1)
        );
        assert_eq!(
            validate_tool_arguments(&tool, &json!({"value": false})).unwrap()["value"],
            json!(0)
        );
    }

    #[test]
    fn coercion_null_to_string() {
        let tool = make_tool(json!({
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"]
        }));
        let result = validate_tool_arguments(&tool, &json!({"value": null})).unwrap();
        assert_eq!(result["value"], json!(""));
    }

    #[test]
    fn coercion_null_to_boolean() {
        let tool = make_tool(json!({
            "type": "object",
            "properties": {"value": {"type": "boolean"}},
            "required": ["value"]
        }));
        let result = validate_tool_arguments(&tool, &json!({"value": null})).unwrap();
        assert_eq!(result["value"], json!(false));
    }

    #[test]
    fn coercion_non_numeric_string_to_number_preserves_original() {
        let tool = make_tool(json!({
            "type": "object",
            "properties": {"value": {"type": "number"}},
            "required": ["value"]
        }));
        let err = validate_tool_arguments(&tool, &json!({"value": "abc"})).unwrap_err();
        assert!(err.to_string().contains("expected number"), "unexpected error: {err}");
    }

    // ------------------------------------------------------------------
    // Union types (anyOf / oneOf)
    // ------------------------------------------------------------------

    #[test]
    fn union_type_string_or_number_accepts_string() {
        let tool = make_tool(json!({
            "type": "object",
            "properties": {
                "value": {"type": ["string", "number"]}
            },
            "required": ["value"]
        }));
        let result = validate_tool_arguments(&tool, &json!({"value": "hello"})).unwrap();
        assert_eq!(result["value"], json!("hello"));
    }

    #[test]
    fn union_type_string_or_number_accepts_number() {
        let tool = make_tool(json!({
            "type": "object",
            "properties": {
                "value": {"type": ["string", "number"]}
            },
            "required": ["value"]
        }));
        let result = validate_tool_arguments(&tool, &json!({"value": 42})).unwrap();
        assert_eq!(result["value"], json!(42));
    }

    #[test]
    fn anyof_schema_accepts_matching_branch() {
        let tool = make_tool(json!({
            "type": "object",
            "properties": {
                "value": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "number"}
                    ]
                }
            },
            "required": ["value"]
        }));
        assert_eq!(
            validate_tool_arguments(&tool, &json!({"value": "hi"})).unwrap()["value"],
            json!("hi")
        );
        assert_eq!(
            validate_tool_arguments(&tool, &json!({"value": 99})).unwrap()["value"],
            json!("99")  // string branch is tried first, coerces 99 to "99"
        );
    }

    #[test]
    fn anyof_schema_rejects_non_matching() {
        let tool = make_tool(json!({
            "type": "object",
            "properties": {
                "value": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "number"}
                    ]
                }
            },
            "required": ["value"]
        }));
        // Arrays cannot coerce to string or number
        let err = validate_tool_arguments(&tool, &json!({"value": [1, 2]})).unwrap_err();
        assert!(err.to_string().contains("anyOf"), "unexpected error: {err}");
    }

    // ------------------------------------------------------------------
    // Additional properties
    // ------------------------------------------------------------------

    #[test]
    fn additional_properties_allowed_by_default() {
        let tool = make_tool(json!({
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"]
        }));
        let args = json!({"name": "alice", "extra": 42});
        let result = validate_tool_arguments(&tool, &args).unwrap();
        assert_eq!(result["name"], json!("alice"));
        assert_eq!(result["extra"], json!(42));
    }

    #[test]
    fn additional_properties_rejected_when_false() {
        let tool = make_tool(json!({
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
            "additionalProperties": false
        }));
        let args = json!({"name": "alice", "extra": 42});
        let err = validate_tool_arguments(&tool, &args).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("unexpected property"), "unexpected error: {msg}");
        assert!(msg.contains("extra"), "should mention the extra field: {msg}");
    }

    // ------------------------------------------------------------------
    // Nested objects
    // ------------------------------------------------------------------

    #[test]
    fn nested_object_validation_passes() {
        let tool = make_tool(json!({
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "number"}
                    },
                    "required": ["name"]
                }
            },
            "required": ["user"]
        }));
        let args = json!({"user": {"name": "bob", "age": 25}});
        let result = validate_tool_arguments(&tool, &args).unwrap();
        assert_eq!(result, args);
    }

    #[test]
    fn nested_object_validation_fails_on_missing_nested_required() {
        let tool = make_tool(json!({
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "number"}
                    },
                    "required": ["name"]
                }
            },
            "required": ["user"]
        }));
        let args = json!({"user": {"age": 25}});
        let err = validate_tool_arguments(&tool, &args).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("required"), "unexpected error: {msg}");
        assert!(msg.contains("user.name"), "should mention nested path: {msg}");
    }

    // ------------------------------------------------------------------
    // Array type
    // ------------------------------------------------------------------

    #[test]
    fn array_type_validates() {
        let tool = make_tool(json!({
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["tags"]
        }));
        let args = json!({"tags": ["a", "b", "c"]});
        let result = validate_tool_arguments(&tool, &args).unwrap();
        assert_eq!(result, args);
    }

    #[test]
    fn array_type_fails_on_wrong_item_type() {
        let tool = make_tool(json!({
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["tags"]
        }));
        // Objects cannot coerce to string, so this should fail
        let args = json!({"tags": [{"nested": true}]});
        let err = validate_tool_arguments(&tool, &args).unwrap_err();
        assert!(err.to_string().contains("expected string"), "unexpected error: {err}");
    }

    #[test]
    fn array_type_fails_on_non_array_input() {
        let tool = make_tool(json!({
            "type": "object",
            "properties": {
                "tags": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["tags"]
        }));
        let args = json!({"tags": "not an array"});
        let err = validate_tool_arguments(&tool, &args).unwrap_err();
        assert!(err.to_string().contains("expected array"), "unexpected error: {err}");
    }

    // ------------------------------------------------------------------
    // Empty object / no required fields
    // ------------------------------------------------------------------

    #[test]
    fn empty_object_validates_with_no_required_fields() {
        let tool = make_tool(json!({
            "type": "object",
            "properties": {
                "optional": {"type": "string"}
            }
        }));
        let args = json!({});
        let result = validate_tool_arguments(&tool, &args).unwrap();
        assert_eq!(result, json!({}));
    }

    #[test]
    fn empty_object_with_no_properties_schema_passes() {
        let tool = make_tool(json!({"type": "object"}));
        let args = json!({});
        let result = validate_tool_arguments(&tool, &args).unwrap();
        assert_eq!(result, json!({}));
    }

    // ------------------------------------------------------------------
    // Type mismatch (no coercion path)
    // ------------------------------------------------------------------

    #[test]
    fn type_mismatch_object_given_array_fails() {
        let tool = make_tool(json!({
            "type": "object",
            "properties": {"data": {"type": "object"}}
        }));
        let args = json!({"data": [1, 2, 3]});
        let err = validate_tool_arguments(&tool, &args).unwrap_err();
        assert!(err.to_string().contains("expected object"), "unexpected error: {err}");
    }

    #[test]
    fn type_mismatch_string_given_object_fails() {
        let tool = make_tool(json!({
            "type": "object",
            "properties": {"value": {"type": "string"}}
        }));
        let args = json!({"value": {"nested": true}});
        let err = validate_tool_arguments(&tool, &args).unwrap_err();
        assert!(err.to_string().contains("expected string"), "unexpected error: {err}");
    }

    // ------------------------------------------------------------------
    // allOf schema
    // ------------------------------------------------------------------

    #[test]
    fn allof_schema_combines_constraints() {
        let tool = make_tool(json!({
            "type": "object",
            "properties": {
                "value": {
                    "allOf": [
                        {"type": "object"},
                        {
                            "type": "object",
                            "properties": {"x": {"type": "number"}},
                            "required": ["x"]
                        }
                    ]
                }
            },
            "required": ["value"]
        }));
        let args_ok = json!({"value": {"x": 1}});
        assert!(validate_tool_arguments(&tool, &args_ok).is_ok());

        let args_bad = json!({"value": {}});
        let err = validate_tool_arguments(&tool, &args_bad).unwrap_err();
        assert!(err.to_string().contains("required"), "unexpected error: {err}");
    }

    // ------------------------------------------------------------------
    // Integer coercion edge cases
    // ------------------------------------------------------------------

    #[test]
    fn integer_coercion_from_string() {
        let tool = make_tool(json!({
            "type": "object",
            "properties": {"n": {"type": "integer"}},
            "required": ["n"]
        }));
        let result = validate_tool_arguments(&tool, &json!({"n": "42"})).unwrap();
        assert_eq!(result["n"], json!(42));
    }

    #[test]
    fn integer_rejects_non_integer_string() {
        let tool = make_tool(json!({
            "type": "object",
            "properties": {"n": {"type": "integer"}},
            "required": ["n"]
        }));
        let err = validate_tool_arguments(&tool, &json!({"n": "3.14"})).unwrap_err();
        assert!(err.to_string().contains("expected integer"), "unexpected error: {err}");
    }

    // ------------------------------------------------------------------
    // Boolean coercion edge cases
    // ------------------------------------------------------------------

    #[test]
    fn boolean_coercion_from_string_true_false() {
        let tool = make_tool(json!({
            "type": "object",
            "properties": {"v": {"type": "boolean"}},
            "required": ["v"]
        }));
        assert_eq!(
            validate_tool_arguments(&tool, &json!({"v": "true"})).unwrap()["v"],
            json!(true)
        );
        assert_eq!(
            validate_tool_arguments(&tool, &json!({"v": "false"})).unwrap()["v"],
            json!(false)
        );
    }

    #[test]
    fn boolean_rejects_arbitrary_string() {
        let tool = make_tool(json!({
            "type": "object",
            "properties": {"v": {"type": "boolean"}},
            "required": ["v"]
        }));
        let err = validate_tool_arguments(&tool, &json!({"v": "yes"})).unwrap_err();
        assert!(err.to_string().contains("expected boolean"), "unexpected error: {err}");
    }

    // ------------------------------------------------------------------
    // Nullable type
    // ------------------------------------------------------------------

    #[test]
    fn null_coercion_to_string_gives_empty() {
        let tool = make_tool(json!({
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"]
        }));
        let result = validate_tool_arguments(&tool, &json!({"name": null})).unwrap();
        assert_eq!(result["name"], json!(""));
    }

    // ------------------------------------------------------------------
    // String to number edge cases
    // ------------------------------------------------------------------

    #[test]
    fn string_number_coercion_preserves_float() {
        let tool = make_tool(json!({
            "type": "object",
            "properties": {"v": {"type": "number"}},
            "required": ["v"]
        }));
        let result = validate_tool_arguments(&tool, &json!({"v": "1.23"})).unwrap();
        assert_eq!(result["v"], json!(1.23));
    }

    #[test]
    fn string_number_coercion_rejects_empty_string() {
        let tool = make_tool(json!({
            "type": "object",
            "properties": {"v": {"type": "number"}},
            "required": ["v"]
        }));
        let err = validate_tool_arguments(&tool, &json!({"v": ""})).unwrap_err();
        assert!(err.to_string().contains("expected number"), "unexpected error: {err}");
    }

    // ------------------------------------------------------------------
    // Additional properties coerced when schema is an object
    // ------------------------------------------------------------------

    #[test]
    fn additional_properties_are_coerced_when_schema_is_object() {
        let tool = make_tool(json!({
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "additionalProperties": {"type": "number"}
        }));
        let args = json!({"name": "alice", "extra": "42"});
        let result = validate_tool_arguments(&tool, &args).unwrap();
        assert_eq!(result["name"], json!("alice"));
        assert_eq!(result["extra"], json!(42));
    }
}
