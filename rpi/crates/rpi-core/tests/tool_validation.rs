//! Tool argument validation parity tests.

use rpi_core::{ToolDefinition, validate_tool_arguments};
use serde_json::{Value, json};

fn tool_with_value_schema(schema: Value) -> ToolDefinition {
    ToolDefinition {
        name: "echo".to_string(),
        description: "Echo tool".to_string(),
        parameters: json!({
            "type": "object",
            "properties": {
                "value": schema
            },
            "required": ["value"]
        }),
    }
}

#[test]
fn validates_and_coerces_plain_json_schema_primitives() {
    let passing_cases = [
        (json!({ "type": "number" }), json!("42"), json!(42)),
        (json!({ "type": "number" }), json!(true), json!(1)),
        (json!({ "type": "number" }), Value::Null, json!(0)),
        (json!({ "type": "integer" }), json!("42"), json!(42)),
        (json!({ "type": "boolean" }), json!("true"), json!(true)),
        (json!({ "type": "boolean" }), json!("false"), json!(false)),
        (json!({ "type": "boolean" }), json!(1), json!(true)),
        (json!({ "type": "boolean" }), json!(0), json!(false)),
        (json!({ "type": "string" }), Value::Null, json!("")),
        (json!({ "type": "string" }), json!(true), json!("true")),
        (json!({ "type": "null" }), json!(""), Value::Null),
        (json!({ "type": "null" }), json!(0), Value::Null),
        (json!({ "type": "null" }), json!(false), Value::Null),
        (
            json!({ "type": ["number", "string"] }),
            json!("1"),
            json!("1"),
        ),
        (
            json!({ "type": ["boolean", "number"] }),
            json!("1"),
            json!(1),
        ),
    ];

    for (schema, input, expected) in passing_cases {
        let tool = tool_with_value_schema(schema);
        let validated = validate_tool_arguments(&tool, &json!({ "value": input })).unwrap();

        assert_eq!(validated, json!({ "value": expected }));
    }
}

#[test]
fn rejects_invalid_plain_json_schema_coercions() {
    let failing_cases = [
        (json!({ "type": "boolean" }), json!("1")),
        (json!({ "type": "boolean" }), json!("0")),
        (json!({ "type": "null" }), json!("null")),
        (json!({ "type": "integer" }), json!("42.1")),
    ];

    for (schema, input) in failing_cases {
        let tool = tool_with_value_schema(schema);
        let err = validate_tool_arguments(&tool, &json!({ "value": input })).unwrap_err();

        assert!(
            err.to_string().contains("Validation failed"),
            "unexpected error: {err}"
        );
    }
}
