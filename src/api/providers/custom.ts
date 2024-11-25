import { Anthropic } from "@anthropic-ai/sdk"
import { Stream as AnthropicStream } from "@anthropic-ai/sdk/streaming"
import { ApiHandlerOptions, ModelInfo } from "../../shared/api"
import { ApiHandler } from "../index"
import { ApiStream } from "../transform/stream"

const customModelInfo: ModelInfo = {
    maxTokens: 4096,
    contextWindow: 16384,
    supportsImages: false,
    supportsComputerUse: true,
    supportsPromptCache: true,
    inputPrice: 0.5,
    outputPrice: 1.5,
    description: "Custom API model with authentication"
}

export class CustomApiHandler implements ApiHandler {
    private options: ApiHandlerOptions
    private client: Anthropic | null = null
    private authToken: string | null = null

    constructor(options: ApiHandlerOptions) {
        this.options = options
        this.authenticate()
    }

    private async authenticate() {
        if (!this.options.customApiKey || !this.options.customApiSecret) {
            throw new Error("Custom API requires both API key and secret for authentication")
        }

        try {
            // Simulate authentication request
            // In a real implementation, this would make an actual API call
            const credentials = {
                apiKey: this.options.customApiKey,
                apiSecret: this.options.customApiSecret
            }
            
            // Simulated token generation
            this.authToken = Buffer.from(JSON.stringify(credentials)).toString('base64')

            // Initialize Anthropic client with custom auth token
            this.client = new Anthropic({
                apiKey: this.authToken
            })
        } catch (error) {
            throw new Error(`Custom API authentication failed: ${error}`)
        }
    }

    async *createMessage(systemPrompt: string, messages: Anthropic.Messages.MessageParam[]): ApiStream {
        if (!this.authToken || !this.client) {
            throw new Error("Not authenticated. Please check your credentials.")
        }

        let stream: AnthropicStream<Anthropic.Beta.PromptCaching.Messages.RawPromptCachingBetaMessageStreamEvent>
        const modelId = this.getModel().id

        try {
            // Use Anthropic's streaming implementation with custom auth
            stream = await this.client.beta.promptCaching.messages.create({
                model: modelId,
                max_tokens: this.getModel().info.maxTokens || 4096,
                temperature: 0,
                system: [{ text: systemPrompt, type: "text", cache_control: { type: "ephemeral" } }],
                messages: messages.map((message, index) => {
                    const userMsgIndices = messages.reduce(
                        (acc, msg, i) => (msg.role === "user" ? [...acc, i] : acc),
                        [] as number[]
                    )
                    const lastUserMsgIndex = userMsgIndices[userMsgIndices.length - 1] ?? -1
                    const secondLastMsgUserIndex = userMsgIndices[userMsgIndices.length - 2] ?? -1

                    if (index === lastUserMsgIndex || index === secondLastMsgUserIndex) {
                        return {
                            ...message,
                            content:
                                typeof message.content === "string"
                                    ? [
                                        {
                                            type: "text",
                                            text: message.content,
                                            cache_control: { type: "ephemeral" },
                                        },
                                    ]
                                    : message.content.map((content, contentIndex) =>
                                        contentIndex === message.content.length - 1
                                            ? { ...content, cache_control: { type: "ephemeral" } }
                                            : content
                                    ),
                        }
                    }
                    return message
                }),
                stream: true,
            }, {
                headers: { "anthropic-beta": "prompt-caching-2024-07-31" }
            })

            for await (const chunk of stream) {
                switch (chunk.type) {
                    case "message_start":
                        const usage = chunk.message.usage
                        yield {
                            type: "usage",
                            inputTokens: usage.input_tokens || 0,
                            outputTokens: usage.output_tokens || 0,
                            cacheWriteTokens: usage.cache_creation_input_tokens || undefined,
                            cacheReadTokens: usage.cache_read_input_tokens || undefined,
                        }
                        break
                    case "message_delta":
                        yield {
                            type: "usage",
                            inputTokens: 0,
                            outputTokens: chunk.usage.output_tokens || 0,
                        }
                        break
                    case "message_stop":
                        break
                    case "content_block_start":
                        switch (chunk.content_block.type) {
                            case "text":
                                if (chunk.index > 0) {
                                    yield {
                                        type: "text",
                                        text: "\n",
                                    }
                                }
                                yield {
                                    type: "text",
                                    text: chunk.content_block.text,
                                }
                                break
                        }
                        break
                    case "content_block_delta":
                        switch (chunk.delta.type) {
                            case "text_delta":
                                yield {
                                    type: "text",
                                    text: chunk.delta.text,
                                }
                                break
                        }
                        break
                    case "content_block_stop":
                        break
                }
            }
        } catch (error) {
            throw new Error(`Custom API request failed: ${error}`)
        }
    }

    getModel(): { id: string; info: ModelInfo } {
        return {
            id: this.options.customModelId ?? "custom-default",
            info: customModelInfo
        }
    }
}
