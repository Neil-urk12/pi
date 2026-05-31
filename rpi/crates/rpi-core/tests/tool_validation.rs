//! Tool argument validation parity tests.

use rpi_core::{ToolCall, ToolDefinition, FunctionCall, validate_tool_arguments, validate_tool_call};
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

// ====================================================================
// Primitive coercion (original 2 tests)
// ====================================================================

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

// ====================================================================
// validate_tool_call
// ====================================================================

#[test]
fn tool_call_finds_matching_tool_and_validates() {
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
fn tool_call_rejects_unknown_tool() {
    let tools = vec![make_tool(json!({"type": "object"}))];
    let call = make_tool_call("nonexistent", json!({}));
    let err = validate_tool_call(&tools, &call).unwrap_err();
    assert!(err.to_string().contains("not found"), "unexpected error: {err}");
}

#[test]
fn tool_call_rejects_malformed_json_arguments() {
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

#[test]
fn tool_call_multiple_tools_selects_correct_one() {
    let tools = vec![
        make_tool(json!({"type": "object", "properties": {"x": {"type": "number"}}})),
        ToolDefinition {
            name: "other".to_string(),
            description: "Other".to_string(),
            parameters: json!({"type": "object", "properties": {"y": {"type": "string"}}, "required": ["y"]}),
        },
    ];
    // "other" requires "y" -- if it picked the wrong tool this would fail
    let call = make_tool_call("other", json!({"y": "hello"}));
    let result = validate_tool_call(&tools, &call).unwrap();
    assert_eq!(result, json!({"y": "hello"}));
}

// ====================================================================
// Nested object validation
// ====================================================================

#[test]
fn nested_object_with_required_fields_passes() {
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
fn nested_object_fails_on_missing_required_field() {
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

#[test]
fn nested_object_extra_properties_rejected_when_false() {
    let tool = make_tool(json!({
        "type": "object",
        "properties": {
            "config": {
                "type": "object",
                "properties": {"host": {"type": "string"}},
                "required": ["host"],
                "additionalProperties": false
            }
        },
        "required": ["config"]
    }));
    let args = json!({"config": {"host": "localhost", "unwanted": true}});
    let err = validate_tool_arguments(&tool, &args).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("unexpected property"), "unexpected error: {msg}");
    assert!(msg.contains("config.unwanted"), "should mention nested extra field: {msg}");
}

#[test]
fn nested_object_extra_properties_allowed_by_default() {
    let tool = make_tool(json!({
        "type": "object",
        "properties": {
            "config": {
                "type": "object",
                "properties": {"host": {"type": "string"}},
                "required": ["host"]
            }
        },
        "required": ["config"]
    }));
    let args = json!({"config": {"host": "localhost", "extra": 42}});
    let result = validate_tool_arguments(&tool, &args).unwrap();
    assert_eq!(result, args);
}

#[test]
fn doubly_nested_object_validates() {
    let tool = make_tool(json!({
        "type": "object",
        "properties": {
            "a": {
                "type": "object",
                "properties": {
                    "b": {
                        "type": "object",
                        "properties": {
                            "c": {"type": "string"}
                        },
                        "required": ["c"]
                    }
                },
                "required": ["b"]
            }
        },
        "required": ["a"]
    }));
    let args = json!({"a": {"b": {"c": "deep"}}});
    assert!(validate_tool_arguments(&tool, &args).is_ok());

    let bad = json!({"a": {"b": {}}});
    let err = validate_tool_arguments(&tool, &bad).unwrap_err();
    assert!(err.to_string().contains("a.b.c"), "should mention deep path: {err}");
}

#[test]
fn nested_object_coerces_types_recursively() {
    let tool = make_tool(json!({
        "type": "object",
        "properties": {
            "user": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "score": {"type": "number"}
                },
                "required": ["name", "score"]
            }
        },
        "required": ["user"]
    }));
    let args = json!({"user": {"name": "bob", "score": "99.5"}});
    let result = validate_tool_arguments(&tool, &args).unwrap();
    assert_eq!(result["user"]["name"], json!("bob"));
    assert_eq!(result["user"]["score"], json!(99.5));
}

// ====================================================================
// Array validation
// ====================================================================

#[test]
fn array_with_correct_item_types_passes() {
    let tool = make_tool(json!({
        "type": "object",
        "properties": {
            "tags": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["tags"]
    }));
    let args = json!({"tags": ["a", "b", "c"]});
    assert_eq!(validate_tool_arguments(&tool, &args).unwrap(), args);
}

#[test]
fn array_with_wrong_item_type_fails() {
    let tool = make_tool(json!({
        "type": "object",
        "properties": {
            "tags": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["tags"]
    }));
    let args = json!({"tags": [{"nested": true}]});
    let err = validate_tool_arguments(&tool, &args).unwrap_err();
    assert!(err.to_string().contains("expected string"), "unexpected error: {err}");
}

#[test]
fn empty_array_passes() {
    let tool = make_tool(json!({
        "type": "object",
        "properties": {
            "items": {"type": "array", "items": {"type": "number"}}
        },
        "required": ["items"]
    }));
    let args = json!({"items": []});
    assert_eq!(validate_tool_arguments(&tool, &args).unwrap(), json!({"items": []}));
}

#[test]
fn non_array_input_fails_array_schema() {
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

#[test]
fn array_item_coercion_works() {
    let tool = make_tool(json!({
        "type": "object",
        "properties": {
            "numbers": {"type": "array", "items": {"type": "number"}}
        },
        "required": ["numbers"]
    }));
    let args = json!({"numbers": ["1", "2", "3.5"]});
    let result = validate_tool_arguments(&tool, &args).unwrap();
    assert_eq!(result["numbers"], json!([1, 2, 3.5]));
}

#[test]
fn array_of_objects_validates_items() {
    let tool = make_tool(json!({
        "type": "object",
        "properties": {
            "users": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"]
                }
            }
        },
        "required": ["users"]
    }));
    let args = json!({"users": [{"name": "a"}, {"name": "b"}]});
    assert!(validate_tool_arguments(&tool, &args).is_ok());

    let bad = json!({"users": [{"name": "a"}, {}]});
    let err = validate_tool_arguments(&tool, &bad).unwrap_err();
    assert!(err.to_string().contains("required"), "unexpected error: {err}");
}

#[test]
fn tuple_items_schema_validates_by_position() {
    let tool = make_tool(json!({
        "type": "object",
        "properties": {
            "pair": {
                "type": "array",
                "items": [
                    {"type": "string"},
                    {"type": "number"}
                ]
            }
        },
        "required": ["pair"]
    }));
    let args = json!({"pair": ["hello", 42]});
    assert!(validate_tool_arguments(&tool, &args).is_ok());

    let bad = json!({"pair": [42, "hello"]});
    let err = validate_tool_arguments(&tool, &bad).unwrap_err();
    assert!(err.to_string().contains("expected number"), "unexpected error: {err}");
}

#[test]
fn tuple_items_coerce_by_position() {
    let tool = make_tool(json!({
        "type": "object",
        "properties": {
            "pair": {
                "type": "array",
                "items": [
                    {"type": "string"},
                    {"type": "number"}
                ]
            }
        },
        "required": ["pair"]
    }));
    let args = json!({"pair": [42, "100"]});
    let result = validate_tool_arguments(&tool, &args).unwrap();
    assert_eq!(result["pair"], json!(["42", 100]));
}

// ====================================================================
// Type mismatch (no coercion path)
// ====================================================================

#[test]
fn object_given_array_fails() {
    let tool = make_tool(json!({
        "type": "object",
        "properties": {"data": {"type": "object"}}
    }));
    let args = json!({"data": [1, 2, 3]});
    let err = validate_tool_arguments(&tool, &args).unwrap_err();
    assert!(err.to_string().contains("expected object"), "unexpected error: {err}");
}

#[test]
fn string_given_object_fails() {
    let tool = make_tool(json!({
        "type": "object",
        "properties": {"value": {"type": "string"}}
    }));
    let args = json!({"value": {"nested": true}});
    let err = validate_tool_arguments(&tool, &args).unwrap_err();
    assert!(err.to_string().contains("expected string"), "unexpected error: {err}");
}

#[test]
fn number_given_array_fails() {
    let tool = make_tool(json!({
        "type": "object",
        "properties": {"value": {"type": "number"}}
    }));
    let args = json!({"value": [1, 2]});
    let err = validate_tool_arguments(&tool, &args).unwrap_err();
    assert!(err.to_string().contains("expected number"), "unexpected error: {err}");
}

#[test]
fn boolean_given_array_fails() {
    let tool = make_tool(json!({
        "type": "object",
        "properties": {"value": {"type": "boolean"}}
    }));
    let args = json!({"value": [true]});
    let err = validate_tool_arguments(&tool, &args).unwrap_err();
    assert!(err.to_string().contains("expected boolean"), "unexpected error: {err}");
}

// ====================================================================
// Additional properties
// ====================================================================

#[test]
fn extra_properties_allowed_by_default() {
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
fn extra_properties_rejected_when_false() {
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

#[test]
fn additional_properties_object_schema_coerces() {
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

// ====================================================================
// Union types (anyOf / oneOf)
// ====================================================================

#[test]
fn union_type_array_accepts_matching_member() {
    let tool = make_tool(json!({
        "type": "object",
        "properties": {
            "value": {"type": ["string", "number"]}
        },
        "required": ["value"]
    }));
    assert_eq!(
        validate_tool_arguments(&tool, &json!({"value": "hello"})).unwrap()["value"],
        json!("hello")
    );
    assert_eq!(
        validate_tool_arguments(&tool, &json!({"value": 42})).unwrap()["value"],
        json!(42)
    );
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
    // string branch tried first, coerces 99 to "99"
    assert_eq!(
        validate_tool_arguments(&tool, &json!({"value": 99})).unwrap()["value"],
        json!("99")
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
    let err = validate_tool_arguments(&tool, &json!({"value": [1, 2]})).unwrap_err();
    assert!(err.to_string().contains("anyOf"), "unexpected error: {err}");
}

#[test]
fn oneof_schema_accepts_matching_branch() {
    let tool = make_tool(json!({
        "type": "object",
        "properties": {
            "value": {
                "oneOf": [
                    {"type": "string"},
                    {"type": "number"}
                ]
            }
        },
        "required": ["value"]
    }));
    let result = validate_tool_arguments(&tool, &json!({"value": "ok"})).unwrap();
    assert_eq!(result["value"], json!("ok"));
}

#[test]
fn oneof_schema_rejects_non_matching() {
    let tool = make_tool(json!({
        "type": "object",
        "properties": {
            "value": {
                "oneOf": [
                    {"type": "string"},
                    {"type": "number"}
                ]
            }
        },
        "required": ["value"]
    }));
    let err = validate_tool_arguments(&tool, &json!({"value": [1, 2]})).unwrap_err();
    assert!(err.to_string().contains("oneOf"), "unexpected error: {err}");
}

#[test]
fn anyof_object_branch_validates_properties() {
    let tool = make_tool(json!({
        "type": "object",
        "properties": {
            "value": {
                "anyOf": [
                    {"type": "string"},
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
    // String branch
    assert_eq!(
        validate_tool_arguments(&tool, &json!({"value": "hi"})).unwrap()["value"],
        json!("hi")
    );
    // Object branch with required field
    assert_eq!(
        validate_tool_arguments(&tool, &json!({"value": {"x": 1}})).unwrap()["value"],
        json!({"x": 1})
    );
    // Object branch missing required field -> fails both branches
    let err = validate_tool_arguments(&tool, &json!({"value": {}})).unwrap_err();
    assert!(err.to_string().contains("anyOf"), "unexpected error: {err}");
}

// ====================================================================
// allOf schema
// ====================================================================

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
    assert!(validate_tool_arguments(&tool, &json!({"value": {"x": 1}})).is_ok());

    let err = validate_tool_arguments(&tool, &json!({"value": {}})).unwrap_err();
    assert!(err.to_string().contains("required"), "unexpected error: {err}");
}

#[test]
fn allof_coerces_through_all_schemas() {
    let tool = make_tool(json!({
        "type": "object",
        "properties": {
            "value": {
                "allOf": [
                    {"type": "object", "properties": {"x": {"type": "number"}}},
                    {"type": "object", "properties": {"y": {"type": "string"}}}
                ]
            }
        },
        "required": ["value"]
    }));
    let args = json!({"value": {"x": "42", "y": 100}});
    let result = validate_tool_arguments(&tool, &args).unwrap();
    assert_eq!(result["value"]["x"], json!(42));
    assert_eq!(result["value"]["y"], json!("100"));
}

// ====================================================================
// Integer coercion edge cases
// ====================================================================

#[test]
fn integer_from_string() {
    let tool = make_tool(json!({
        "type": "object",
        "properties": {"n": {"type": "integer"}},
        "required": ["n"]
    }));
    assert_eq!(
        validate_tool_arguments(&tool, &json!({"n": "42"})).unwrap()["n"],
        json!(42)
    );
}

#[test]
fn integer_rejects_float_string() {
    let tool = make_tool(json!({
        "type": "object",
        "properties": {"n": {"type": "integer"}},
        "required": ["n"]
    }));
    let err = validate_tool_arguments(&tool, &json!({"n": "3.14"})).unwrap_err();
    assert!(err.to_string().contains("expected integer"), "unexpected error: {err}");
}

#[test]
fn integer_from_bool() {
    let tool = make_tool(json!({
        "type": "object",
        "properties": {"n": {"type": "integer"}},
        "required": ["n"]
    }));
    assert_eq!(
        validate_tool_arguments(&tool, &json!({"n": true})).unwrap()["n"],
        json!(1)
    );
    assert_eq!(
        validate_tool_arguments(&tool, &json!({"n": false})).unwrap()["n"],
        json!(0)
    );
}

#[test]
fn integer_from_null() {
    let tool = make_tool(json!({
        "type": "object",
        "properties": {"n": {"type": "integer"}},
        "required": ["n"]
    }));
    assert_eq!(
        validate_tool_arguments(&tool, &json!({"n": null})).unwrap()["n"],
        json!(0)
    );
}

// ====================================================================
// Boolean coercion edge cases
// ====================================================================

#[test]
fn boolean_from_string_true_false() {
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

#[test]
fn boolean_from_number_0_and_1() {
    let tool = make_tool(json!({
        "type": "object",
        "properties": {"v": {"type": "boolean"}},
        "required": ["v"]
    }));
    assert_eq!(
        validate_tool_arguments(&tool, &json!({"v": 1})).unwrap()["v"],
        json!(true)
    );
    assert_eq!(
        validate_tool_arguments(&tool, &json!({"v": 0})).unwrap()["v"],
        json!(false)
    );
}

#[test]
fn boolean_rejects_non_0_1_number() {
    let tool = make_tool(json!({
        "type": "object",
        "properties": {"v": {"type": "boolean"}},
        "required": ["v"]
    }));
    let err = validate_tool_arguments(&tool, &json!({"v": 2})).unwrap_err();
    assert!(err.to_string().contains("expected boolean"), "unexpected error: {err}");
}

#[test]
fn boolean_from_null() {
    let tool = make_tool(json!({
        "type": "object",
        "properties": {"v": {"type": "boolean"}},
        "required": ["v"]
    }));
    assert_eq!(
        validate_tool_arguments(&tool, &json!({"v": null})).unwrap()["v"],
        json!(false)
    );
}

// ====================================================================
// Number coercion edge cases
// ====================================================================

#[test]
fn number_from_float_string() {
    let tool = make_tool(json!({
        "type": "object",
        "properties": {"v": {"type": "number"}},
        "required": ["v"]
    }));
    assert_eq!(
        validate_tool_arguments(&tool, &json!({"v": "1.23"})).unwrap()["v"],
        json!(1.23)
    );
}

#[test]
fn number_rejects_empty_string() {
    let tool = make_tool(json!({
        "type": "object",
        "properties": {"v": {"type": "number"}},
        "required": ["v"]
    }));
    let err = validate_tool_arguments(&tool, &json!({"v": ""})).unwrap_err();
    assert!(err.to_string().contains("expected number"), "unexpected error: {err}");
}

#[test]
fn number_from_bool_true_false() {
    let tool = make_tool(json!({
        "type": "object",
        "properties": {"v": {"type": "number"}},
        "required": ["v"]
    }));
    assert_eq!(
        validate_tool_arguments(&tool, &json!({"v": true})).unwrap()["v"],
        json!(1)
    );
    assert_eq!(
        validate_tool_arguments(&tool, &json!({"v": false})).unwrap()["v"],
        json!(0)
    );
}

#[test]
fn number_from_null() {
    let tool = make_tool(json!({
        "type": "object",
        "properties": {"v": {"type": "number"}},
        "required": ["v"]
    }));
    assert_eq!(
        validate_tool_arguments(&tool, &json!({"v": null})).unwrap()["v"],
        json!(0)
    );
}

#[test]
fn number_preserves_integer_value() {
    let tool = make_tool(json!({
        "type": "object",
        "properties": {"v": {"type": "number"}},
        "required": ["v"]
    }));
    let result = validate_tool_arguments(&tool, &json!({"v": 42})).unwrap();
    // Should stay as integer, not become 42.0
    assert_eq!(result["v"], json!(42));
}

// ====================================================================
// String coercion edge cases
// ====================================================================

#[test]
fn string_from_null_gives_empty() {
    let tool = make_tool(json!({
        "type": "object",
        "properties": {"v": {"type": "string"}},
        "required": ["v"]
    }));
    assert_eq!(
        validate_tool_arguments(&tool, &json!({"v": null})).unwrap()["v"],
        json!("")
    );
}

#[test]
fn string_from_number() {
    let tool = make_tool(json!({
        "type": "object",
        "properties": {"v": {"type": "string"}},
        "required": ["v"]
    }));
    assert_eq!(
        validate_tool_arguments(&tool, &json!({"v": 42})).unwrap()["v"],
        json!("42")
    );
}

#[test]
fn string_from_bool() {
    let tool = make_tool(json!({
        "type": "object",
        "properties": {"v": {"type": "string"}},
        "required": ["v"]
    }));
    assert_eq!(
        validate_tool_arguments(&tool, &json!({"v": true})).unwrap()["v"],
        json!("true")
    );
}

// ====================================================================
// Null coercion edge cases
// ====================================================================

#[test]
fn null_from_empty_string() {
    let tool = make_tool(json!({
        "type": "object",
        "properties": {"v": {"type": "null"}},
        "required": ["v"]
    }));
    assert_eq!(
        validate_tool_arguments(&tool, &json!({"v": ""})).unwrap()["v"],
        Value::Null
    );
}

#[test]
fn null_from_zero() {
    let tool = make_tool(json!({
        "type": "object",
        "properties": {"v": {"type": "null"}},
        "required": ["v"]
    }));
    assert_eq!(
        validate_tool_arguments(&tool, &json!({"v": 0})).unwrap()["v"],
        Value::Null
    );
}

#[test]
fn null_from_false() {
    let tool = make_tool(json!({
        "type": "object",
        "properties": {"v": {"type": "null"}},
        "required": ["v"]
    }));
    assert_eq!(
        validate_tool_arguments(&tool, &json!({"v": false})).unwrap()["v"],
        Value::Null
    );
}

#[test]
fn null_preserves_non_zero_values() {
    let tool = make_tool(json!({
        "type": "object",
        "properties": {"v": {"type": "null"}},
        "required": ["v"]
    }));
    // "hello" is not empty-string, not zero, not false -> stays as-is
    let err = validate_tool_arguments(&tool, &json!({"v": "hello"})).unwrap_err();
    assert!(err.to_string().contains("expected null"), "unexpected error: {err}");
}

// ====================================================================
// Empty objects and no required fields
// ====================================================================

#[test]
fn empty_object_with_optional_fields_passes() {
    let tool = make_tool(json!({
        "type": "object",
        "properties": {
            "optional": {"type": "string"}
        }
    }));
    assert_eq!(
        validate_tool_arguments(&tool, &json!({})).unwrap(),
        json!({})
    );
}

#[test]
fn empty_object_with_no_properties_passes() {
    let tool = make_tool(json!({"type": "object"}));
    assert_eq!(
        validate_tool_arguments(&tool, &json!({})).unwrap(),
        json!({})
    );
}

#[test]
fn bare_object_not_object_fails() {
    let tool = make_tool(json!({"type": "object"}));
    let err = validate_tool_arguments(&tool, &json!("not an object")).unwrap_err();
    assert!(err.to_string().contains("expected object"), "unexpected error: {err}");
}

// ====================================================================
// Multiple required fields
// ====================================================================

#[test]
fn multiple_required_fields_all_present_passes() {
    let tool = make_tool(json!({
        "type": "object",
        "properties": {
            "a": {"type": "string"},
            "b": {"type": "number"},
            "c": {"type": "boolean"}
        },
        "required": ["a", "b", "c"]
    }));
    let args = json!({"a": "x", "b": 1, "c": true});
    assert_eq!(validate_tool_arguments(&tool, &args).unwrap(), args);
}

#[test]
fn multiple_required_fields_one_missing_fails() {
    let tool = make_tool(json!({
        "type": "object",
        "properties": {
            "a": {"type": "string"},
            "b": {"type": "number"},
            "c": {"type": "boolean"}
        },
        "required": ["a", "b", "c"]
    }));
    let args = json!({"a": "x", "c": true});
    let err = validate_tool_arguments(&tool, &args).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("required"), "unexpected error: {msg}");
    assert!(msg.contains("b"), "should mention missing field b: {msg}");
}

#[test]
fn multiple_required_fields_all_missing_reports_all() {
    let tool = make_tool(json!({
        "type": "object",
        "properties": {
            "a": {"type": "string"},
            "b": {"type": "number"}
        },
        "required": ["a", "b"]
    }));
    let args = json!({});
    let err = validate_tool_arguments(&tool, &args).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("a"), "should mention missing field a: {msg}");
    assert!(msg.contains("b"), "should mention missing field b: {msg}");
}

// ====================================================================
// No schema type (untyped properties)
// ====================================================================

#[test]
fn untyped_property_accepts_anything() {
    let tool = make_tool(json!({
        "type": "object",
        "properties": {
            "data": {}
        },
        "required": ["data"]
    }));
    // All these should pass -- no type constraint
    assert!(validate_tool_arguments(&tool, &json!({"data": "str"})).is_ok());
    assert!(validate_tool_arguments(&tool, &json!({"data": 42})).is_ok());
    assert!(validate_tool_arguments(&tool, &json!({"data": [1]})).is_ok());
    assert!(validate_tool_arguments(&tool, &json!({"data": {"k": "v"}})).is_ok());
    assert!(validate_tool_arguments(&tool, &json!({"data": null})).is_ok());
    assert!(validate_tool_arguments(&tool, &json!({"data": true})).is_ok());
}

// ====================================================================
// Error message format
// ====================================================================

#[test]
fn error_includes_tool_name_and_received_args() {
    let tool = make_tool(json!({
        "type": "object",
        "properties": {"x": {"type": "number"}},
        "required": ["x"]
    }));
    let args = json!({"y": "wrong"});
    let err = validate_tool_arguments(&tool, &args).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("test_tool"), "should include tool name: {msg}");
    assert!(msg.contains("Validation failed"), "should say validation failed: {msg}");
}

// ====================================================================
// Whitespace in string-to-number coercion
// ====================================================================

#[test]
fn whitespace_only_string_rejects_number() {
    let tool = make_tool(json!({
        "type": "object",
        "properties": {"v": {"type": "number"}},
        "required": ["v"]
    }));
    let err = validate_tool_arguments(&tool, &json!({"v": "   "})).unwrap_err();
    assert!(err.to_string().contains("expected number"), "unexpected error: {err}");
}

#[test]
fn whitespace_only_string_rejects_integer() {
    let tool = make_tool(json!({
        "type": "object",
        "properties": {"v": {"type": "integer"}},
        "required": ["v"]
    }));
    let err = validate_tool_arguments(&tool, &json!({"v": "   "})).unwrap_err();
    assert!(err.to_string().contains("expected integer"), "unexpected error: {err}");
}

// ====================================================================
// Negative number coercion
// ====================================================================

#[test]
fn negative_number_from_string() {
    let tool = make_tool(json!({
        "type": "object",
        "properties": {"v": {"type": "number"}},
        "required": ["v"]
    }));
    assert_eq!(
        validate_tool_arguments(&tool, &json!({"v": "-3.14"})).unwrap()["v"],
        json!(-3.14)
    );
}

#[test]
fn negative_integer_from_string() {
    let tool = make_tool(json!({
        "type": "object",
        "properties": {"v": {"type": "integer"}},
        "required": ["v"]
    }));
    assert_eq!(
        validate_tool_arguments(&tool, &json!({"v": "-42"})).unwrap()["v"],
        json!(-42)
    );
}
