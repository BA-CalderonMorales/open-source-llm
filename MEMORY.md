# Development Guidelines for All LLMs/agents

## Core Philosophy

**TEST-DRIVEN DEVELOPMENT IS NON-NEGOTIABLE.** Every single line of production code must be written in response to a failing test. No exceptions. This is not a suggestion or a preference - it is the fundamental practice that enables all other principles in this document.

I follow Test-Driven Development (TDD) with a strong emphasis on behavior-driven testing and functional programming principles. All work should be done in small, incremental changes that maintain a working state throughout development.

## Quick Reference

**Key Principles:**

- Write tests first (TDD)
- Test behavior, not implementation
- No `any` types or type assertions
- Immutable data only
- Small, pure functions
- TypeScript strict mode always
- Use real schemas/types in tests, never redefine them

## Repository Overview

This project is an experimental reimplementation of the DeepSeek large language model.
Key components:

- `inference/` – Python code for running a simplified Transformer model.
- `mobile/` – Rust library that quantizes weights and exposes a C-compatible API for mobile apps.
- `inference-re/` – Minimal Rust version of the inference stack used mainly for tests.

### Current State

- Basic model forward pass and token generation utilities exist for both Python and Rust.
- Mobile crate supports saving and loading quantized weights and simple tokenization.
- Continuous integration only runs Rust tests; Python tests are optional.
- The project does not use Node.js. No `package.json` or `package-lock.json`
  files exist, so skip `npm ci`, `npm test`, `npm run typecheck`, and
  `npm run build`.

### Limitations

- No GPU support in CI; heavy dependencies like PyTorch must be installed manually.
- The Python model lacks unit tests and advanced features from the original DeepSeek implementation.
- FFI bindings are minimal and primarily intended for demonstration.

### Roadmap Ideas

- Improve Python coverage with lightweight CPU-friendly tests.
- Expand the mobile API with additional tokenization helpers.
- Investigate simple optimizations (e.g., mmap weights) for faster startup.

**Preferred Tools:**

- **Language**: TypeScript (strict mode)
- **Testing**: Jest/Vitest + React Testing Library
- **State Management**: Prefer immutable patterns

## Testing Principles

### Behavior-Driven Testing

- **No "unit tests"** - this term is not helpful. Tests should verify expected behavior, treating implementation as a black box
- Test through the public API exclusively - internals should be invisible to tests
- No 1:1 mapping between test files and implementation files
- Tests that examine internal implementation details are wasteful and should be avoided
- **Coverage targets**: 100% coverage should be expected at all times, but these tests must ALWAYS be based on business behaviour, not implementation details
- Tests must document expected business behaviour

### Testing Tools

- **Jest** or **Vitest** for testing frameworks
- **React Testing Library** for React components
- **MSW (Mock Service Worker)** for API mocking when needed
- All test code must follow the same TypeScript strict mode rules as production code

### Test Organization

```
src/
  features/
    payment/
      payment-processor.ts
      payment-validator.ts
      payment-processor.test.ts // The validator is an implementation detail. Validation is fully covered, but by testing the expected business behaviour, treating the validation code itself as an implementation detail
```

### Test Data Pattern

Use factory functions with optional overrides for test data:

```typescript
const getMockPaymentPostPaymentRequest = (
  overrides?: Partial<PostPaymentsRequestV3>
): PostPaymentsRequestV3 => {
  return {
    CardAccountId: "1234567890123456",
    Amount: 100,
    Source: "Web",
    AccountStatus: "Normal",
    LastName: "Doe",
    DateOfBirth: "1980-01-01",
    PayingCardDetails: {
      Cvv: "123",
      Token: "token",
    },
    AddressDetails: getMockAddressDetails(),
    Brand: "Visa",
    ...overrides,
  };
};

const getMockAddressDetails = (
  overrides?: Partial<AddressDetails>
): AddressDetails => {
  return {
    HouseNumber: "123",
    HouseName: "Test House",
    AddressLine1: "Test Address Line 1",
    AddressLine2: "Test Address Line 2",
    City: "Test City",
    ...overrides,
  };
};
```

Key principles:

- Always return complete objects with sensible defaults
- Accept optional `Partial<T>` overrides
- Build incrementally - extract nested object factories as needed
- Compose factories for complex objects
- Consider using a test data builder pattern for very complex objects

## TypeScript Guidelines

### Strict Mode Requirements

```json
// tsconfig.json
{
  "compilerOptions": {
    "strict": true,
    "noImplicitAny": true,
    "strictNullChecks": true,
    "strictFunctionTypes": true,
    "strictBindCallApply": true,
    "strictPropertyInitialization": true,
    "noImplicitThis": true,
    "alwaysStrict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noImplicitReturns": true,
    "noFallthroughCasesInSwitch": true
  }
}
```

- **No `any`** - ever. Use `unknown` if type is truly unknown
- **No type assertions** (`as SomeType`) unless absolutely necessary with clear justification
- **No `@ts-ignore`** or `@ts-expect-error` without explicit explanation
- These rules apply to test code as well as production code

### Type Definitions

- **Prefer `type` over `interface`** in all cases
- Use explicit typing where it aids clarity, but leverage inference where appropriate
- Utilize utility types effectively (`Pick`, `Omit`, `Partial`, `Required`, etc.)
- Create domain-specific types (e.g., `UserId`, `PaymentId`) for type safety
- Use Zod or any other [Standard Schema](https://standardschema.dev/) compliant schema library to create types, by creating schemas first

```typescript
// Good
type UserId = string & { readonly brand: unique symbol };
type PaymentAmount = number & { readonly brand: unique symbol };

// Avoid
type UserId = string;
type PaymentAmount = number;
```

#### Schema-First Development with Zod

Always define your schemas first, then derive types from them:

```typescript
import { z } from "zod";

// Define schemas first - these provide runtime validation
const AddressDetailsSchema = z.object({
  houseNumber: z.string(),
  houseName: z.string().optional(),
  addressLine1: z.string().min(1),
  addressLine2: z.string().optional(),
  city: z.string().min(1),
  postcode: z.string().regex(/^[A-Z]{1,2}\d[A-Z\d]? ?\d[A-Z]{2}$/i),
});

const PayingCardDetailsSchema = z.object({
  cvv: z.string().regex(/^\d{3,4}$/),
  token: z.string().min(1),
});

const PostPaymentsRequestV3Schema = z.object({
  cardAccountId: z.string().length(16),
  amount: z.number().positive(),
  source: z.enum(["Web", "Mobile", "API"]),
  accountStatus: z.enum(["Normal", "Restricted", "Closed"]),
  lastName: z.string().min(1),
  dateOfBirth: z.string().regex(/^\d{4}-\d{2}-\d{2}$/),
  payingCardDetails: PayingCardDetailsSchema,
  addressDetails: AddressDetailsSchema,
  brand: z.enum(["Visa", "Mastercard", "Amex"]),
});

// Derive types from schemas
type AddressDetails = z.infer<typeof AddressDetailsSchema>;
type PayingCardDetails = z.infer<typeof PayingCardDetailsSchema>;
type PostPaymentsRequestV3 = z.infer<typeof PostPaymentsRequestV3Schema>;

// Use schemas at runtime boundaries
export const parsePaymentRequest = (data: unknown): PostPaymentsRequestV3 => {
  return PostPaymentsRequestV3Schema.parse(data);
};

// Example of schema composition for complex domains
const BaseEntitySchema = z.object({
  id: z.string().uuid(),
  createdAt: z.date(),
  updatedAt: z.date(),
});

const CustomerSchema = BaseEntitySchema.extend({
  email: z.string().email(),
  tier: z.enum(["standard", "premium", "enterprise"]),
  creditLimit: z.number().positive(),
});

type Customer = z.infer<typeof CustomerSchema>;
```

#### Schema Usage in Tests

**CRITICAL**: Tests must use real schemas and types from the main project, not redefine their own.

```typescript
// ❌ WRONG - Defining schemas in test files
const ProjectSchema = z.object({
  id: z.string(),
  workspaceId: z.string(),
  ownerId: z.string().nullable(),
  name: z.string(),
  createdAt: z.coerce.date(),
  updatedAt: z.coerce.date(),
});

// ✅ CORRECT - Import schemas from the shared schema package
import { ProjectSchema, type Project } from "@your-org/schemas";
```

**Why this matters:**

- **Type Safety**: Ensures tests use the same types as production code
- **Consistency**: Changes to schemas automatically propagate to tests
- **Maintainability**: Single source of truth for data structures
- **Prevents Drift**: Tests can't accidentally diverge from real schemas

**Implementation:**

- All domain schemas should be exported from a shared schema package or module
- Test files should import schemas from the shared location
- If a schema isn't exported yet, add it to the exports rather than duplicating it
- Mock data factories should use the real types derived from real schemas

```typescript
// ✅ CORRECT - Test factories using real schemas
import { ProjectSchema, type Project } from "@your-org/schemas";

const getMockProject = (overrides?: Partial<Project>): Project => {
  const baseProject = {
    id: "proj_123",
    workspaceId: "ws_456",
    ownerId: "user_789",
    name: "Test Project",
    createdAt: new Date(),
    updatedAt: new Date(),
  };

  const projectData = { ...baseProject, ...overrides };

  // Validate against real schema to catch type mismatches
  return ProjectSchema.parse(projectData);
};
```

## Code Style

### Functional Programming

I follow a "functional light" approach:

- **No data mutation** - work with immutable data structures
- **Pure functions** wherever possible
- **Composition** as the primary mechanism for code reuse
- Avoid heavy FP abstractions (no need for complex monads or pipe/compose patterns) unless there is a clear advantage to using them
- Use array methods (`map`, `filter`, `reduce`) over imperative loops

#### Examples of Functional Patterns

```typescript
// Good - Pure function with immutable updates
const applyDiscount = (order: Order, discountPercent: number): Order => {
  return {
    ...order,
    items: order.items.map((item) => ({
      ...item,
      price: item.price * (1 - discountPercent / 100),
    })),
    totalPrice: order.items.reduce(
      (sum, item) => sum + item.price * (1 - discountPercent / 100),
      0
    ),
  };
};

// Good - Composition over complex logic
const processOrder = (order: Order): ProcessedOrder => {
  return pipe(
    order,
    validateOrder,
    applyPromotions,
    calculateTax,
    assignWarehouse
  );
};

// When heavy FP abstractions ARE appropriate:
// - Complex async flows that benefit from Task/IO types
// - Error handling chains that benefit from Result/Either types
// Example with Result type for complex error handling:
type Result<T, E = Error> =
  | { success: true; data: T }
  | { success: false; error: E };

const chainPaymentOperations = (
  payment: Payment
): Result<Receipt, PaymentError> => {
  return pipe(
    validatePayment(payment),
    chain(authorizePayment),
    chain(capturePayment),
    map(generateReceipt)
  );
};
```

### Code Structure

- **No nested if/else statements** - use early returns, guard clauses, or composition
- **Avoid deep nesting** in general (max 2 levels)
- Keep functions small and focused on a single responsibility
- Prefer flat, readable code over clever abstractions

### Naming Conventions

- **Functions**: `camelCase`, verb-based (e.g., `calculateTotal`, `validatePayment`)
- **Types**: `PascalCase` (e.g., `PaymentRequest`, `UserProfile`)
- **Constants**: `UPPER_SNAKE_CASE` for true constants, `camelCase` for configuration
- **Files**: `kebab-case.ts` for all TypeScript files
- **Test files**: `*.test.ts` or `*.spec.ts`

### No Comments in Code

Code should be self-documenting through clear naming and structure. Comments indicate that the code itself is not clear enough.

```typescript
// Avoid: Comments explaining what the code does
const calculateDiscount = (price: number, customer: Customer): number => {
  // Check if customer is premium
  if (customer.tier === "premium") {
    // Apply 20% discount for premium customers
    return price * 0.8;
  }
  // Regular customers get 10% discount
  return price * 0.9;
};

// Good: Self-documenting code with clear names
const PREMIUM_DISCOUNT_MULTIPLIER = 0.8;
const STANDARD_DISCOUNT_MULTIPLIER = 0.9;

const isPremiumCustomer = (customer: Customer): boolean => {
  return customer.tier === "premium";
};

const calculateDiscount = (price: number, customer: Customer): number => {
  const discountMultiplier = isPremiumCustomer(customer)
    ? PREMIUM_DISCOUNT_MULTIPLIER
    : STANDARD_DISCOUNT_MULTIPLIER;

  return price * discountMultiplier;
};

// Avoid: Complex logic with comments
const processPayment = (payment: Payment): ProcessedPayment => {
  // First validate the payment
  if (!validatePayment(payment)) {
    throw new Error("Invalid payment");
  }

  // Check if we need to apply 3D secure
  if (payment.amount > 100 && payment.card.type === "credit") {
    // Apply 3D secure for credit cards over £100
    const securePayment = apply3DSecure(payment);
    // Process the secure payment
    return executePayment(securePayment);
  }

  // Process the payment
  return executePayment(payment);
};

// Good: Extract to well-named functions
const requires3DSecure = (payment: Payment): boolean => {
  const SECURE_PAYMENT_THRESHOLD = 100;
  return (
    payment.amount > SECURE_PAYMENT_THRESHOLD && payment.card.type === "credit"
  );
};

const processPayment = (payment: Payment): ProcessedPayment => {
  if (!validatePayment(payment)) {
    throw new PaymentValidationError("Invalid payment");
  }

  const securedPayment = requires3DSecure(payment)
    ? apply3DSecure(payment)
    : payment;

  return executePayment(securedPayment);
};
```

**Exception**: JSDoc comments for public APIs are acceptable when generating documentation, but the code should still be self-explanatory without them.

### Prefer Options Objects

Use options objects for function parameters as the default pattern. Only use positional parameters when there's a clear, compelling reason (e.g., single-parameter pure functions, well-established conventions like `map(item => item.value)`).

```typescript
// Avoid: Multiple positional parameters
const createPayment = (
  amount: number,
  currency: string,
  cardId: string,
  customerId: string,
  description?: string,
  metadata?: Record<string, unknown>,
  idempotencyKey?: string
): Payment => {
  // implementation
};

// Calling it is unclear
const payment = createPayment(
  100,
  "GBP",
  "card_123",
  "cust_456",
  undefined,
  { orderId: "order_789" },
  "key_123"
);

// Good: Options object with clear property names
type CreatePaymentOptions = {
  amount: number;
  currency: string;
  cardId: string;
  customerId: string;
  description?: string;
  metadata?: Record<string, unknown>;
  idempotencyKey?: string;
};

const createPayment = (options: CreatePaymentOptions): Payment => {
  const {
    amount,
    currency,
    cardId,
    customerId,
    description,
    metadata,
    idempotencyKey,
  } = options;

  // implementation
};

// Clear and readable at call site
const payment = createPayment({
  amount: 100,
  currency: "GBP",
  cardId: "card_123",
  customerId: "cust_456",
  metadata: { orderId: "order_789" },
  idempotencyKey: "key_123",
});

// Avoid: Boolean flags as parameters
const fetchCustomers = (
  includeInactive: boolean,
  includePending: boolean,
  includeDeleted: boolean,
  sortByDate: boolean
): Customer[] => {
  // implementation
};

// Confusing at call site
const customers = fetchCustomers(true, false, false, true);

// Good: Options object with clear intent
type FetchCustomersOptions = {
  includeInactive?: boolean;
  includePending?: boolean;
  includeDeleted?: boolean;
  sortBy?: "date" | "name" | "value";
};

const fetchCustomers = (options: FetchCustomersOptions = {}): Customer[] => {
  const {
    includeInactive = false,
    includePending = false,
    includeDeleted = false,
    sortBy = "name",
  } = options;

  // implementation
};

// Self-documenting at call site
const customers = fetchCustomers({
  includeInactive: true,
  sortBy: "date",
});

// Good: Configuration objects for complex operations
type ProcessOrderOptions = {
  order: Order;
  shipping: {
    method: "standard" | "express" | "overnight";
    address: Address;
  };
  payment: {
    method: PaymentMethod;
    saveForFuture?: boolean;
  };
  promotions?: {
    codes?: string[];
    autoApply?: boolean;
  };
};

const processOrder = (options: ProcessOrderOptions): ProcessedOrder => {
  const { order, shipping, payment, promotions = {} } = options;

  // Clear access to nested options
  const orderWithPromotions = promotions.autoApply
    ? applyAvailablePromotions(order)
    : order;

  return executeOrder({
    ...orderWithPromotions,
    shippingMethod: shipping.method,
    paymentMethod: payment.method,
  });
};

// Acceptable: Single parameter for simple transforms
const double = (n: number): number => n * 2;
const getName = (user: User): string => user.name;

// Acceptable: Well-established patterns
const numbers = [1, 2, 3];
const doubled = numbers.map((n) => n * 2);
const users = fetchUsers();
const names = users.map((user) => user.name);
```

**Guidelines for options objects:**

- Default to options objects unless there's a specific reason not to
- Always use for functions with optional parameters
- Destructure options at the start of the function for clarity
- Provide sensible defaults using destructuring
- Keep related options grouped (e.g., all shipping options together)
- Consider breaking very large options objects into nested groups

**When positional parameters are acceptable:**

- Single-parameter pure functions
- Well-established functional patterns (map, filter, reduce callbacks)
- Mathematical operations where order is conventional

## Development Workflow

### TDD Process - THE FUNDAMENTAL PRACTICE

**CRITICAL**: TDD is not optional. Every feature, every bug fix, every change MUST follow this process:

Follow Red-Green-Refactor strictly:

1. **Red**: Write a failing test for the desired behavior. NO PRODUCTION CODE until you have a failing test.
2. **Green**: Write the MINIMUM code to make the test pass. Resist the urge to write more than needed.
3. **Refactor**: Assess the code for improvement opportunities. If refactoring would add value, clean up the code while keeping tests green. If the code is already clean and expressive, move on.

**Common TDD Violations to Avoid:**

- Writing production code without a failing test first
- Writing multiple tests before making the first one pass
- Writing more production code than needed to pass the current test
- Skipping the refactor assessment step when code could be improved
- Adding functionality "while you're there" without a test driving it

**Remember**: If you're typing production code and there isn't a failing test demanding that code, you're not doing TDD.

#### TDD Example Workflow

```typescript
// Step 1: Red - Start with the simplest behavior
describe("Order processing", () => {
  it("should calculate total with shipping cost", () => {
    const order = createOrder({
      items: [{ price: 30, quantity: 1 }],
      shippingCost: 5.99,
    });

    const processed = processOrder(order);

    expect(processed.total).toBe(35.99);
    expect(processed.shippingCost).toBe(5.99);
  });
});

// Step 2: Green - Minimal implementation
const processOrder = (order: Order): ProcessedOrder => {
  const itemsTotal = order.items.reduce(
    (sum, item) => sum + item.price * item.quantity,
    0
  );

  return {
    ...order,
    shippingCost: order.shippingCost,
    total: itemsTotal + order.shippingCost,
  };
};

// Step 3: Red - Add test for free shipping behavior
describe("Order processing", () => {
  it("should calculate total with shipping cost", () => {
    // ... existing test
  });

  it("should apply free shipping for orders over \u00a350", () => {
    const order = createOrder({
      items: [{ price: 60, quantity: 1 }],
      shippingCost: 5.99,
    });

    const processed = processOrder(order);

    expect(processed.shippingCost).toBe(0);
    expect(processed.total).toBe(60);
  });
});

// Step 4: Green - NOW we can add the conditional because both paths are tested
const processOrder = (order: Order): ProcessedOrder => {
  const itemsTotal = order.items.reduce(
    (sum, item) => sum + item.price * item.quantity,
    0
  );

  const shippingCost = itemsTotal > 50 ? 0 : order.shippingCost;

  return {
    ...order,
    shippingCost,
    total: itemsTotal + shippingCost,
  };
};

// Step 5: Add edge case tests to ensure 100% behavior coverage
describe("Order processing", () => {
  // ... existing tests

  it("should charge shipping for orders exactly at \u00a350", () => {
    const order = createOrder({
      items: [{ price: 50, quantity: 1 }],
      shippingCost: 5.99,
    });

    const processed = processOrder(order);

    expect(processed.shippingCost).toBe(5.99);
    expect(processed.total).toBe(55.99);
  });
});

// Step 6: Refactor - Extract constants and improve readability
const FREE_SHIPPING_THRESHOLD = 50;

const calculateItemsTotal = (items: OrderItem[]): number => {
  return items.reduce((sum, item) => sum + item.price * item.quantity, 0);
};

const qualifiesForFreeShipping = (itemsTotal: number): boolean => {
  return itemsTotal > FREE_SHIPPING_THRESHOLD;
};

const processOrder = (order: Order): ProcessedOrder => {
  const itemsTotal = calculateItemsTotal(order.items);
  const shippingCost = qualifiesForFreeShipping(itemsTotal)
    ? 0
    : order.shippingCost;

  return {
    ...order,
    shippingCost,
    total: itemsTotal + shippingCost,
  };
};
```

### Refactoring - The Critical Third Step

Evaluating refactoring opportunities is not optional - it's the third step in the TDD cycle. After achieving a green state and committing your work, you MUST assess whether the code can be improved. However, only refactor if there's clear value - if the code is already clean and expresses intent well, move on to the next test.

#### What is Refactoring?

Refactoring means changing the internal structure of code without changing its external behavior. The public API remains unchanged, all tests continue to pass, but the code becomes cleaner, more maintainable, or more efficient. Remember: only refactor when it genuinely improves the code - not all code needs refactoring.

#### When to Refactor

- **Always assess after green**: Once tests pass, before moving to the next test, evaluate if refactoring would add value
- **When you see duplication**: But understand what duplication really means (see DRY below)
- **When names could be clearer**: Variable names, function names, or type names that don't clearly express intent
- **When structure could be simpler**: Complex conditional logic, deeply nested code, or long functions
- **When patterns emerge**: After implementing several similar features, useful abstractions may become apparent

**Remember**: Not all code needs refactoring. If the code is already clean, expressive, and well-structured, commit and move on.

#### Refactoring Guidelines

##### 1. Commit Before Refactoring

Always commit your working code before starting any refactoring. This gives you a safe point to return to:

```bash
git add .
git commit -m "feat: add payment validation"
# Now safe to refactor
```

##### 2. Look for Useful Abstractions Based on Semantic Meaning

Create abstractions only when code shares the same semantic meaning and purpose. Don't abstract based on structural similarity alone - **duplicate code is far cheaper than the wrong abstraction**.

```typescript
// Similar structure, DIFFERENT semantic meaning - DO NOT ABSTRACT
const validatePaymentAmount = (amount: number): boolean => {
  return amount > 0 && amount <= 10000;
};

const validateTransferAmount = (amount: number): boolean => {
  return amount > 0 && amount <= 10000;
};

// These might have the same structure today, but they represent different
// business concepts that will likely evolve independently.
// Payment limits might change based on fraud rules.
// Transfer limits might change based on account type.
// Abstracting them couples unrelated business rules.

// Similar structure, SAME semantic meaning - SAFE TO ABSTRACT
const formatUserDisplayName = (firstName: string, lastName: string): string => {
  return `${firstName} ${lastName}`.trim();
};

const formatCustomerDisplayName = (
  firstName: string,
  lastName: string
): string => {
  return `${firstName} ${lastName}`.trim();
};

const formatEmployeeDisplayName = (
  firstName: string,
  lastName: string
): string => {
  return `${firstName} ${lastName}`.trim();
};

// These all represent the same concept: "how we format a person's name for display"
// They share semantic meaning, not just structure
const formatPersonDisplayName = (
  firstName: string,
  lastName: string
): string => {
  return `${firstName} ${lastName}`.trim();
};

// Replace all call sites throughout the codebase:
// Before:
// const userLabel = formatUserDisplayName(user.firstName, user.lastName);
// const customerName = formatCustomerDisplayName(customer.firstName, customer.lastName);
// const employeeTag = formatEmployeeDisplayName(employee.firstName, employee.lastName);

// After:
// const userLabel = formatPersonDisplayName(user.firstName, user.lastName);
// const customerName = formatPersonDisplayName(customer.firstName, customer.lastName);
// const employeeTag = formatPersonDisplayName(employee.firstName, employee.lastName);

// Then remove the original functions as they're no longer needed
```

**Questions to ask before abstracting:**

- Do these code blocks represent the same concept or different concepts that happen to look similar?
- If the business rules for one change, should the others change too?
- Would a developer reading this abstraction understand why these things are grouped together?
- Am I abstracting based on what the code IS (structure) or what the code MEANS (semantics)?

**Remember**: It's much easier to create an abstraction later when the semantic relationship becomes clear than to undo a bad abstraction that couples unrelated concepts.

##### 3. Understanding DRY - It's About Knowledge, Not Code

DRY (Don't Repeat Yourself) is about not duplicating **knowledge** in the system, not about eliminating all code that looks similar.

```typescript
// This is NOT a DRY violation - different knowledge despite similar code
const validateUserAge = (age: number): boolean => {
  return age >= 18 && age <= 100;
};

const validateProductRating = (rating: number): boolean => {
  return rating >= 1 && rating <= 5;
};

const validateYearsOfExperience = (years: number): boolean => {
  return years >= 0 && years <= 50;
};

// These functions have similar structure (checking numeric ranges), but they
// represent completely different business rules:
// - User age has legal requirements (18+) and practical limits (100)
// - Product ratings follow a 1-5 star system
// - Years of experience starts at 0 with a reasonable upper bound
// Abstracting them would couple unrelated business concepts and make future
// changes harder. What if ratings change to 1-10? What if legal age changes?

// Another example of code that looks similar but represents different knowledge:
const formatUserDisplayName = (user: User): string => {
  return `${user.firstName} ${user.lastName}`.trim();
};

const formatAddressLine = (address: Address): string => {
  return `${address.street} ${address.number}`.trim();
};

const formatCreditCardLabel = (card: CreditCard): string => {
  return `${card.type} ${card.lastFourDigits}`.trim();
};

// Despite the pattern "combine two strings with space and trim", these represent
// different domain concepts with different future evolution paths

// This IS a DRY violation - same knowledge in multiple places
class Order {
  calculateTotal(): number {
    const itemsTotal = this.items.reduce((sum, item) => sum + item.price, 0);
    const shippingCost = itemsTotal > 50 ? 0 : 5.99; // Knowledge duplicated!
    return itemsTotal + shippingCost;
  }
}

class OrderSummary {
  getShippingCost(itemsTotal: number): number {
    return itemsTotal > 50 ? 0 : 5.99; // Same knowledge!
  }
}

class ShippingCalculator {
  calculate(orderAmount: number): number {
    if (orderAmount > 50) return 0; // Same knowledge again!
    return 5.99;
  }
}

// Refactored - knowledge in one place
const FREE_SHIPPING_THRESHOLD = 50;
const STANDARD_SHIPPING_COST = 5.99;

const calculateShippingCost = (itemsTotal: number): number => {
  return itemsTotal > FREE_SHIPPING_THRESHOLD ? 0 : STANDARD_SHIPPING_COST;
};

// Now all classes use the single source of truth
class Order {
  calculateTotal(): number {
    const itemsTotal = this.items.reduce((sum, item) => sum + item.price, 0);
    return itemsTotal + calculateShippingCost(itemsTotal);
  }
}
```

##### 4. Maintain External APIs During Refactoring

Refactoring must never break existing consumers of your code:

```typescript
// Original implementation
export const processPayment = (payment: Payment): ProcessedPayment => {
  // Complex logic all in one function
  if (payment.amount <= 0) {
    throw new Error("Invalid amount");
  }

  if (payment.amount > 10000) {
    throw new Error("Amount too large");
  }

  // ... 50 more lines of validation and processing

  return result;
};

// Refactored - external API unchanged, internals improved
export const processPayment = (payment: Payment): ProcessedPayment => {
  validatePaymentAmount(payment.amount);
  validatePaymentMethod(payment.method);

  const authorizedPayment = authorizePayment(payment);
  const capturedPayment = capturePayment(authorizedPayment);

  return generateReceipt(capturedPayment);
};

// New internal functions - not exported
const validatePaymentAmount = (amount: number): void => {
  if (amount <= 0) {
    throw new Error("Invalid amount");
  }

  if (amount > 10000) {
    throw new Error("Amount too large");
  }
};

// Tests continue to pass without modification because external API unchanged
```

##### 5. Verify and Commit After Refactoring

**CRITICAL**: After every refactoring:

1. Run all tests - they must pass without modification
2. Run static analysis (linting, type checking) - must pass
3. Commit the refactoring separately from feature changes

```bash
# After refactoring
cargo test --workspace  # All Rust tests must pass
pytest                  # All Python tests must pass

# Only then commit
git add .
git commit -m "refactor: extract payment validation helpers"
```

#### Refactoring Checklist

Before considering refactoring complete, verify:

- [ ] The refactoring actually improves the code (if not, don't refactor)
- [ ] All tests still pass without modification
- [ ] All static analysis tools pass (linting, type checking)
- [ ] No new public APIs were added (only internal ones)
- [ ] Code is more readable than before
- [ ] Any duplication removed was duplication of knowledge, not just code
- [ ] No speculative abstractions were created
- [ ] The refactoring is committed separately from feature changes

#### Example Refactoring Session

```typescript
// After getting tests green with minimal implementation:
describe("Order processing", () => {
  it("calculates total with items and shipping", () => {
    const order = { items: [{ price: 30 }, { price: 20 }], shipping: 5 };
    expect(calculateOrderTotal(order)).toBe(55);
  });

  it("applies free shipping over \u00a350", () => {
    const order = { items: [{ price: 30 }, { price: 25 }], shipping: 5 };
    expect(calculateOrderTotal(order)).toBe(55);
  });
});

// Green implementation (minimal):
const calculateOrderTotal = (order: Order): number => {
  const itemsTotal = order.items.reduce((sum, item) => sum + item.price, 0);
  const shipping = itemsTotal > 50 ? 0 : order.shipping;
  return itemsTotal + shipping;
};

// Commit the working version
// git commit -m "feat: implement order total calculation with free shipping"

// Assess refactoring opportunities:
// - The variable names could be clearer
// - The free shipping threshold is a magic number
// - The calculation logic could be extracted for clarity
// These improvements would add value, so proceed with refactoring:

const FREE_SHIPPING_THRESHOLD = 50;

const calculateItemsTotal = (items: OrderItem[]): number => {
  return items.reduce((sum, item) => sum + item.price, 0);
};

const calculateShipping = (
  baseShipping: number,
  itemsTotal: number
): number => {
  return itemsTotal > FREE_SHIPPING_THRESHOLD ? 0 : baseShipping;
};

const calculateOrderTotal = (order: Order): number => {
  const itemsTotal = calculateItemsTotal(order.items);
  const shipping = calculateShipping(order.shipping, itemsTotal);
  return itemsTotal + shipping;
};

// Run tests - they still pass!
// Run linting - all clean!
// Run type checking - no errors!

// Now commit the refactoring
// git commit -m "refactor: extract order total calculation helpers"
```

##### Example: When NOT to Refactor

```typescript
// After getting this test green:
describe("Discount calculation", () => {
  it("should apply 10% discount", () => {
    const originalPrice = 100;
    const discountedPrice = applyDiscount(originalPrice, 0.1);
    expect(discountedPrice).toBe(90);
  });
});

// Green implementation:
const applyDiscount = (price: number, discountRate: number): number => {
  return price * (1 - discountRate);
};

// Assess refactoring opportunities:
// - Code is already simple and clear
// - Function name clearly expresses intent
// - Implementation is a straightforward calculation
// - No magic numbers or unclear logic
// Conclusion: No refactoring needed. This is fine as-is.

// Commit and move to the next test
// git commit -m "feat: add discount calculation"
```

### Commit Guidelines

- Each commit should represent a complete, working change
- Use conventional commits format:
  ```
  feat: add payment validation
  fix: correct date formatting in payment processor
  refactor: extract payment validation logic
  test: add edge cases for payment validation
  ```
- Include test changes with feature changes in the same commit

### Pull Request Standards

- Every PR must have all tests passing
- All linting and quality checks must pass
- Work in small increments that maintain a working state
- PRs should be focused on a single feature or fix
- Include description of the behavior change, not implementation details

### Pre-commit Hooks

Use `pre-commit` to run tests and commitlint on every commit. Install hooks with:

```bash
pre-commit install
```

Commits will be blocked if tests fail or commit messages do not follow the conventional format.

## Working with Claude

### Expectations

When working with my code:

1. **ALWAYS FOLLOW TDD** - No production code without a failing test. This is not negotiable.
2. **Think deeply** before making any edits
3. **Understand the full context** of the code and requirements
4. **Ask clarifying questions** when requirements are ambiguous
5. **Think from first principles** - don't make assumptions
6. **Assess refactoring after every green** - Look for opportunities to improve code structure, but only refactor if it adds value
7. **Keep project docs current** - update them whenever you introduce meaningful changes

### Code Changes

When suggesting or making changes:

- **Start with a failing test** - always. No exceptions.
- After making tests pass, always assess refactoring opportunities (but only refactor if it adds value)
- After refactoring, verify all tests and static analysis pass, then commit
- Respect the existing patterns and conventions
- Maintain test coverage for all behavior changes
- Keep changes small and incremental
- Ensure all TypeScript strict mode requirements are met
- Provide rationale for significant design decisions

**If you find yourself writing production code without a failing test, STOP immediately and write the test first.**

### Communication

- Be explicit about trade-offs in different approaches
- Explain the reasoning behind significant design decisions
- Flag any deviations from these guidelines with justification
- Suggest improvements that align with these principles
- When unsure, ask for clarification rather than assuming

## Example Patterns

### Error Handling

Use Result types or early returns:

```typescript
// Good - Result type pattern
type Result<T, E = Error> =
  | { success: true; data: T }
  | { success: false; error: E };

const processPayment = (
  payment: Payment
): Result<ProcessedPayment, PaymentError> => {
  if (!isValidPayment(payment)) {
    return { success: false, error: new PaymentError("Invalid payment") };
  }

  if (!hasSufficientFunds(payment)) {
    return { success: false, error: new PaymentError("Insufficient funds") };
  }

  return { success: true, data: executePayment(payment) };
};

// Also good - early returns with exceptions
const processPayment = (payment: Payment): ProcessedPayment => {
  if (!isValidPayment(payment)) {
    throw new PaymentError("Invalid payment");
  }

  if (!hasSufficientFunds(payment)) {
    throw new PaymentError("Insufficient funds");
  }

  return executePayment(payment);
};
```

### Testing Behavior

```typescript
// Good - tests behavior through public API
describe("PaymentProcessor", () => {
  it("should decline payment when insufficient funds", () => {
    const payment = getMockPaymentPostPaymentRequest({ Amount: 1000 });
    const account = getMockAccount({ Balance: 500 });

    const result = processPayment(payment, account);

    expect(result.success).toBe(false);
    expect(result.error.message).toBe("Insufficient funds");
  });

  it("should process valid payment successfully", () => {
    const payment = getMockPaymentPostPaymentRequest({ Amount: 100 });
    const account = getMockAccount({ Balance: 500 });

    const result = processPayment(payment, account);

    expect(result.success).toBe(true);
    expect(result.data.remainingBalance).toBe(400);
  });
});

// Avoid - testing implementation details
describe("PaymentProcessor", () => {
  it("should call checkBalance method", () => {
    // This tests implementation, not behavior
  });
});
```

#### Achieving 100% Coverage Through Business Behavior

Example showing how validation code gets 100% coverage without testing it directly:

```typescript
// payment-validator.ts (implementation detail)
export const validatePaymentAmount = (amount: number): boolean => {
  return amount > 0 && amount <= 10000;
};

export const validateCardDetails = (card: PayingCardDetails): boolean => {
  return /^\d{3,4}$/.test(card.cvv) && card.token.length > 0;
};

// payment-processor.ts (public API)
export const processPayment = (
  request: PaymentRequest
): Result<Payment, PaymentError> => {
  // Validation is used internally but not exposed
  if (!validatePaymentAmount(request.amount)) {
    return { success: false, error: new PaymentError("Invalid amount") };
  }

  if (!validateCardDetails(request.payingCardDetails)) {
    return { success: false, error: new PaymentError("Invalid card details") };
  }

  // Process payment...
  return { success: true, data: executedPayment };
};

// payment-processor.test.ts
describe("Payment processing", () => {
  // These tests achieve 100% coverage of validation code
  // without directly testing the validator functions

  it("should reject payments with negative amounts", () => {
    const payment = getMockPaymentPostPaymentRequest({ amount: -100 });
    const result = processPayment(payment);

    expect(result.success).toBe(false);
    expect(result.error.message).toBe("Invalid amount");
  });

  it("should reject payments exceeding maximum amount", () => {
    const payment = getMockPaymentPostPaymentRequest({ amount: 10001 });
    const result = processPayment(payment);

    expect(result.success).toBe(false);
    expect(result.error.message).toBe("Invalid amount");
  });

  it("should reject payments with invalid CVV format", () => {
    const payment = getMockPaymentPostPaymentRequest({
      payingCardDetails: { cvv: "12", token: "valid-token" },
    });
    const result = processPayment(payment);

    expect(result.success).toBe(false);
    expect(result.error.message).toBe("Invalid card details");
  });

  it("should process valid payments successfully", () => {
    const payment = getMockPaymentPostPaymentRequest({
      amount: 100,
      payingCardDetails: { cvv: "123", token: "valid-token" },
    });
    const result = processPayment(payment);

    expect(result.success).toBe(true);
    expect(result.data.status).toBe("completed");
  });
});
```

### React Component Testing

```typescript
// Good - testing user-visible behavior
describe("PaymentForm", () => {
  it("should show error when submitting invalid amount", async () => {
    render(<PaymentForm />);

    const amountInput = screen.getByLabelText("Amount");
    const submitButton = screen.getByRole("button", { name: "Submit Payment" });

    await userEvent.type(amountInput, "-100");
    await userEvent.click(submitButton);

    expect(screen.getByText("Amount must be positive")).toBeInTheDocument();
  });
});
```

## Common Patterns to Avoid

### Anti-patterns

```typescript
// Avoid: Mutation
const addItem = (items: Item[], newItem: Item) => {
  items.push(newItem); // Mutates array
  return items;
};

// Prefer: Immutable update
const addItem = (items: Item[], newItem: Item): Item[] => {
  return [...items, newItem];
};

// Avoid: Nested conditionals
if (user) {
  if (user.isActive) {
    if (user.hasPermission) {
      // do something
    }
  }
}

// Prefer: Early returns
if (!user || !user.isActive || !user.hasPermission) {
  return;
}
// do something

// Avoid: Large functions
const processOrder = (order: Order) => {
  // 100+ lines of code
};

// Prefer: Composed small functions
const processOrder = (order: Order) => {
  const validatedOrder = validateOrder(order);
  const pricedOrder = calculatePricing(validatedOrder);
  const finalOrder = applyDiscounts(pricedOrder);
  return submitOrder(finalOrder);
};
```

## Resources and References

- [TypeScript Handbook](https://www.typescriptlang.org/docs/handbook/intro.html)
- [Testing Library Principles](https://testing-library.com/docs/guiding-principles)
- [Kent C. Dodds Testing JavaScript](https://testingjavascript.com/)
- [Functional Programming in TypeScript](https://gcanti.github.io/fp-ts/)

## Summary

The key is to write clean, testable, functional code that evolves through small, safe increments. Every change should be driven by a test that describes the desired behavior, and the implementation should be the simplest thing that makes that test pass. When in doubt, favor simplicity and readability over cleverness.
# Software Engineering Laws: Expanded Interpretations & Examples

This document provides detailed interpretations and real-world examples—both successful and cautionary—for classic software engineering laws, drawn from decades of experience across startups, open-source projects, FAANG, GPU engineering (Nvidia/AMD), and AAA game studios (Ubisoft, Xbox, PlayStation).

---

## Index

- Murphy’s Law  
- Brook’s Law  
- Hofstadter’s Law  
- Conway’s Law  
- Postel’s Law  
- Pareto Principle  
- The Peter Principle  
- Kerckhoffs’s Principle  
- Linus’s Law  
- Moore’s Law  
- Wirth’s Law  
- Ninety-ninety Rule  
- Knuth’s Optimization Principle  
- Norvig’s Law  

---

## Murphy’s Law

> **“Anything that can go wrong, will go wrong.”**

**Interpretation:**  
No matter how bullet-proof your design, unexpected failures—bugs, infrastructure outages, security incidents—will happen. Defensive coding, exhaustive testing (TDD/MDD), robust CI/CD, and disaster-recovery plans are your shield.

**Good Examples:**  
1. **Startup**: A two-person fintech startup adopted feature flags and automated rollback in their CI pipeline. When a corner-case bug in payments surfaced in production, they disabled the feature flag instantly and fixed it within hours—averting financial loss.  
2. **Open Source**: The maintainers of a cross-platform library ran CI tests on Linux, macOS, Windows, and FreeBSD. When a compiler update broke one target, the CI alerts caught it before any downstream projects were impacted.

**Bad Examples:**  
1. **FAANG**: A service team pushed a database schema change without backward-compatibility checks. When cache nodes failed to migrate, the entire global service was unavailable for four hours.  
2. **AAA Gaming**: A console title shipped with an untested physics edge case. Players encountering the rare scenario saw the game crash near end-boss fights, triggering negative reviews and emergency patches.

---

## Brook’s Law

> **“Adding manpower to a late software project makes it later.”**

**Interpretation:**  
Onboarding new team members consumes time—mentoring, code reviews, communication overhead. Simply hiring more developers rarely accelerates a delayed project; better to refine scope, improve processes, or refactor architecture.

**Good Examples:**  
1. **AAA Studio (Ubisoft)**: Rather than hiring ten junior designers mid-sprint, leadership brought in two senior producers to streamline the pipeline, reprioritize features, and unblock existing engineers—accelerating delivery.  
2. **Open Source**: Instead of recruiting more maintainers when a release lagged, the project improved contributor documentation and CI guidance, reducing ramp-up time and getting volunteers up to speed faster.

**Bad Examples:**  
1. **FAANG**: A datacenter team added six engineers to fix a lagging monitoring service. Time spent teaching code standards and architecture nuances pushed the deadline further out.  
2. **Early-Stage Startup**: Founders panicked, hired four fresh graduates to meet a launch date. The ramp-up and hand-holding required derailed core feature work, delaying release by months.

---

## Hofstadter’s Law

> **“It always takes longer than you expect, even when you take into account Hofstadter’s Law.”**

**Interpretation:**  
Estimating complex tasks is inherently flawed. Even with experience, unforeseen complications—third-party quirks, architectural edge cases, shifting requirements—inflate timelines. Always pad your estimates.

**Good Examples:**  
1. **Nvidia Driver Team**: Engineers built in a 25% time buffer for a major driver rewrite. When hardware quirks surfaced, the project still finished on schedule.  
2. **FAANG Agile Squad**: Adopted sprint planning with built-in “slack” stories. When an integration issue ate up half a sprint, they still met deliverables by deferring low-priority items.

**Bad Examples:**  
1. **Startup MVP**: Team promised a working prototype in four weeks. Unexpected API rate limits and refactoring took six months—burning through runway.  
2. **Game Engine Rewrite**: A AAA studio estimated a quarter for a refactor; under-estimated deep coupling, resulting in repeated delays and feature freezes.

---

## Conway’s Law

> **“Organizations which design systems are constrained to produce designs which are copies of the communication structures of these organizations.”**

**Interpretation:**  
Your team topology shapes your architecture. Siloed teams yield siloed code; cross-functional squads yield cohesive services. Align team boundaries with desired system modules to minimize handoffs.

**Good Examples:**  
1. **FAANG Microservices**: Cross-functional “service teams” owned each microservice end-to-end—frontend, backend, infrastructure—delivering independent deploys with minimal coordination.  
2. **AAA Game Feature Teams**: Designers, artists, and gameplay engineers formed stable pods around major features (e.g., AI systems), leading to tighter integration and faster iteration.

**Bad Examples:**  
1. **Legacy Monolith**: Frontend, backend, and DB teams worked in silos. Every UI tweak required three separate pull requests and cross-team meetings, causing weeks of delay.  
2. **Small Startup**: Developers split by language (Java vs. JavaScript). API mismatches and duplicated logic plagued the codebase, as feature boundaries didn’t match team skills.

---

## Postel’s Law

> **“Be conservative in what you send, be liberal in what you accept.”**

**Interpretation:**  
Design your outputs strictly (predictable formats), but parse inputs flexibly (tolerant of variations). This robustness principle underpins resilient networks and APIs—too lax in output or too strict in input both cause failures.

**Good Examples:**  
1. **Open Source HTTP Client**: Strictly formats headers on send but tolerates missing or reordered headers in responses, improving compatibility with varied servers.  
2. **Game Engine Plugin API**: Engine writes standardized JSON logs but accepts plugin configs with minor formatting differences, reducing integration friction.

**Bad Examples:**  
1. **FAANG Analytics Service**: Accepted any malformed telemetry until storage filled with garbage data, corrupting dashboards and BI reports.  
2. **Early-Stage Startup Form**: Over-permissive input sanitization allowed SQL injection attacks when users submitted unexpected characters.

---

## Pareto Principle

> **“80% of consequences come from 20% of causes.”**

**Interpretation:**  
A small fraction of your code typically produces most bugs or carries most performance weight. Focus efforts on that critical “20%” to maximize impact—whether fixing bugs, refactoring hot spots, or writing docs.

**Good Examples:**  
1. **FAANG DB Team**: Profiled queries, optimized the top 20% slowest, and saw overall load-time drop by 70%.  
2. **Open Source Library**: Identified the 20% of API methods used by 80% of clients; improved their docs and stability, reducing support issues.

**Bad Examples:**  
1. **AAA UI Polish**: Spent weeks animating secondary menus (used <10% by players) while core gameplay suffered frame-rate issues.  
2. **Startup Feature Bloat**: Built niche integrations for a tiny customer subset, neglecting core UX, resulting in poor adoption.

---

## The Peter Principle

> **“In a hierarchy, every employee tends to rise to his level of incompetence.”**

**Interpretation:**  
Engineers promoted for coding skill often struggle when moved into management roles. Without parallel technical and managerial tracks, organizations lose both coding capacity and leadership quality.

**Good Examples:**  
1. **FAANG Dual Ladder**: Engineers choose between an “IC path” or “Manager path.” Top individual contributors advance technically without forced people management.  
2. **AAA Studio Mentorship**: Senior devs promoted to “Tech Lead” received coaching in leadership and people skills—reducing turnover and performance dips.

**Bad Examples:**  
1. **Startup**: A star coder was promoted to CTO without management training; project delivery suffered due to poor delegation and communication.  
2. **Large Corp**: Ineffective manager promoted repeatedly; team morale plummeted and attrition spiked.

---

## Kerckhoffs’s Principle

> **“A cryptosystem should be secure even if everything about the system, except the key, is public knowledge.”**

**Interpretation:**  
Never depend on secrecy of algorithms—security must rest solely on key confidentiality. Public scrutiny (open algorithms) strengthens confidence.

**Good Examples:**  
1. **Open Source Crypto Library**: Published algorithm specs, focused audits on key storage and management, earning enterprise trust.  
2. **FAANG Service Encryption**: Used industry-standard AES; all code public, key material stored in hardware security modules (HSMs).

**Bad Examples:**  
1. **Early-Stage Startup**: Rolled a proprietary cipher believing obscurity would protect them; attackers reversed it in days.  
2. **AAA DRM System**: Obfuscated key handling in game client; hackers extracted keys and cracked content within weeks.

---

## Linus’s Law

> **“Given enough eyeballs, all bugs are shallow.”**

**Interpretation:**  
Open code review and a large contributor base catch defects faster. Public, transparent development yields higher quality through collective scrutiny.

**Good Examples:**  
1. **Linux Kernel**: Thousands of developers and companies identify and fix bugs within days of code submissions.  
2. **Open Game Engine**: Welcomed community pull requests and bug reports; stability improved dramatically between versions.

**Bad Examples:**  
1. **Closed Beta**: A select tester group missed UI freezes that only appeared under varied hardware; bugs reached full release.  
2. **FAANG Internal Tool**: Restricting repo access to one team delayed bug discovery until after customer rollout.

---

## Moore’s Law

> **“Transistor count (and hence raw computing power) doubles approximately every 18–24 months.”**

**Interpretation:**  
Rapid hardware improvements tempt developers to add features or layers rather than optimize. But reliance on ever-cheaper cycles can backfire if hardware gains slow.

**Good Examples:**  
1. **Nvidia GPU Dev**: Leveraged each new GPU gen for real-time ray tracing features, pushing game visuals forward without crippled performance.  
2. **Cloud Startup**: Scaled compute-heavy AI workloads cost-effectively by shifting to next-gen instances, reducing time-to-market.

**Bad Examples:**  
1. **Software Bloat**: Each release adds abstraction layers assuming Moore’s Law will cover the perf hit—eventually systems grind to a crawl.  
2. **AAA Porting**: Ported a PC engine to console without optimization, expecting console hardware to “catch up,” resulting in poor frame rates.

---

## Wirth’s Law

> **“Software gets slower faster than hardware gets faster.”**

**Interpretation:**  
Unchecked feature creep and heavy frameworks can negate Moore’s Law gains. Engineers must stay vigilant: profile, trim, and optimize critical paths.

**Good Examples:**  
1. **Embedded Systems**: Team wrote performance-critical routines in C, hand-optimized inner loops, keeping firmware snappy on limited hardware.  
2. **Game Asset Pipeline**: Automated texture compression and streaming, minimizing memory footprint and load times.

**Bad Examples:**  
1. **FAANG Microservices Sprawl**: Service mesh, side-cars, and proxies piled on, increasing request latency beyond hardware improvements.  
2. **Web Startup**: Adopted heavy front-end frameworks without profiling; initial page load times on mobile ballooned to 10+ seconds.

---

## Ninety-ninety Rule

> **“The first 90% of the code takes 10% of the time; the remaining 10% takes the other 90% of the time.”**

**Interpretation:**  
Early development seems fast—core features come together quickly. Polishing, edge cases, testing, and deployment often consume most of the schedule.

**Good Examples:**  
1. **FAANG Release Process**: Allocated equal time for alpha, beta, and release-candidate cycles, ensuring final polish didn’t get squeezed.  
2. **AAA QA Pipeline**: Built multiple playtest phases into the calendar; final 10% of bug fixes and performance tuning had dedicated sprints.

**Bad Examples:**  
1. **Startup MVP Rush**: Skipped QA at the end to hit “launch” date. Customers found critical bugs; reputation suffered.  
2. **Open Source Release**: Maintainers focused on new features, ignored final integration testing—users encountered show-stopping regressions.

---

## Knuth’s Optimization Principle

> **“Premature optimization is the root of all evil.”**

**Interpretation:**  
Write clear, correct code first. Only after profiling identifies true bottlenecks should you optimize. Otherwise you waste time on unimportant paths.

**Good Examples:**  
1. **GPU Shader Team**: Profiled frame-render loop, then optimized only the hottest shader variants—gained 30% perf with minimal complexity.  
2. **Web Scale Startup**: Used APM tooling to find slow DB queries, refactored just those queries, speeding up page loads by 50%.

**Bad Examples:**  
1. **AAA Engine Rewrite**: Spent months hand-tuning rendering subsystems before game mechanics were defined—ultimate game delayed.  
2. **FAANG Feature Sprint**: Developer rewrote JSON parser for tiny speed gain, introducing subtle bugs and delaying feature shipping.

---

## Norvig’s Law

> **“Any technology that surpasses 50% penetration will never double again.”**

**Interpretation:**  
Once a product, platform, or language reaches mainstream saturation, further growth slows dramatically. Recognize maturity and adjust strategy accordingly.

**Good Examples:**  
1. **Search Engine Product**: At 60% market share, team pivoted to adjacent services (maps, cloud) instead of chasing further search growth.  
2. **AAA Game Engine**: Recognized core engine had peaked in adoption; focused on new middleware offerings rather than pushing new engine sales.

**Bad Examples:**  
1. **Social Startup**: After reaching 55% of its target demographic, poured all resources into growth hacks that yielded diminishing returns.  
2. **GPU Architecture**: Attempted to double transistor count annually beyond physical limits, leading to skyrocketing costs and thermal issues.

---

## Conclusion

Understanding—and applying—these laws with real-world context helps you anticipate pitfalls, make informed trade-offs, and guide teams through complexity. Whether you’re in a two-person startup or a global FAANG organization, these principles will serve as your compass.

---
