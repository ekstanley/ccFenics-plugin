/-!
# Error Code Distinctness

Proves that all 13 error types in errors.py have distinct error_code strings.
-/

namespace DolfinxProofs

/-- Enumeration of all 13 error codes from errors.py. -/
inductive ErrorCode where
  | dolfinxMcpError
  | noActiveMesh
  | meshNotFound
  | functionSpaceNotFound
  | functionNotFound
  | invalidUflExpression
  | solverError
  | duplicateName
  | dolfinxApiError
  | fileIoError
  | preconditionViolated
  | postconditionViolated
  | invariantViolated
  deriving DecidableEq, Repr

/-- Maps each ErrorCode variant to its Python error_code string. -/
def errorCodeString : ErrorCode → String
  | .dolfinxMcpError       => "DOLFINX_MCP_ERROR"
  | .noActiveMesh          => "NO_ACTIVE_MESH"
  | .meshNotFound          => "MESH_NOT_FOUND"
  | .functionSpaceNotFound => "FUNCTION_SPACE_NOT_FOUND"
  | .functionNotFound      => "FUNCTION_NOT_FOUND"
  | .invalidUflExpression  => "INVALID_UFL_EXPRESSION"
  | .solverError           => "SOLVER_ERROR"
  | .duplicateName         => "DUPLICATE_NAME"
  | .dolfinxApiError       => "DOLFINX_API_ERROR"
  | .fileIoError           => "FILE_IO_ERROR"
  | .preconditionViolated  => "PRECONDITION_VIOLATED"
  | .postconditionViolated => "POSTCONDITION_VIOLATED"
  | .invariantViolated     => "INVARIANT_VIOLATED"

/-- The mapping is injective: distinct error types have distinct codes. -/
theorem errorCodeString_injective (a : ErrorCode) (b : ErrorCode)
    (h : errorCodeString a = errorCodeString b) : a = b := by
  cases a <;> cases b <;> simp_all [errorCodeString]

/-- All 13 error code strings are pairwise distinct. -/
theorem allErrorCodeStrings_nodup :
    ([ "DOLFINX_MCP_ERROR", "NO_ACTIVE_MESH", "MESH_NOT_FOUND",
       "FUNCTION_SPACE_NOT_FOUND", "FUNCTION_NOT_FOUND",
       "INVALID_UFL_EXPRESSION", "SOLVER_ERROR", "DUPLICATE_NAME",
       "DOLFINX_API_ERROR", "FILE_IO_ERROR", "PRECONDITION_VIOLATED",
       "POSTCONDITION_VIOLATED", "INVARIANT_VIOLATED" ] : List String).Nodup := by
  native_decide

end DolfinxProofs
